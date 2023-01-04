// Start of Initial Setup

#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <chrono>
#include <random>
#include <functional>
#include "eiquadprog.hpp"

using namespace std;
using namespace Eigen;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
    return std::max(lower, std::min(n, upper));
}

int k = 0;

MatrixXf parameters;

// Read and write CSV files

class CSVData
{
    public:
    MatrixXf data;
    string filename;

    CSVData(string filename_, MatrixXf data_)
    {
        filename = filename_;
        data = data_;
    }

    void writeToCSVfile()
    {
        ofstream file(filename.c_str());
        file << data.format(CSVFormat);
        file.close();
    }

    MatrixXf readFromCSVfile()
    {
        vector<float> matrixEntries;
        ifstream matrixDataFile(filename);
        string matrixRowString;
        string matrixEntry;
        int matrixRowNumber = 0;
    
        while (getline(matrixDataFile, matrixRowString))
        {
            stringstream matrixRowStringStream(matrixRowString);
            while (getline(matrixRowStringStream, matrixEntry, ','))
            {
                matrixEntries.push_back(stod(matrixEntry));
            }
            matrixRowNumber++;
        }
        
        return Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
    }
};

// End of Initial Setup

float v_cruise = 0.00;

// Pedestrian Model assuming an exponentially distributed inter-arrival time and gaussian crossing speed

class PedestrianModel
{
public:
    int num_pedestrians;
    VectorXf times;
    VectorXf speeds;
    VectorXf positions;
    float w, Ts, t_end, t_start, lambda_;
    float pd_v;
    float pd_sigma;
    float stop_probability;
    VectorXf random_status;
    VectorXf real_status;
    float o;

    PedestrianModel()
    {
        int jT = k*(k < 5) + (k-5)*((k >= 5) & (k < 10)) + 2*(k >= 10); 
        int jT_ = 1*((k >= 5) & (k < 10));
        int jo = (k-10)*((k >= 10) & (k < 15)) + 2*((k < 10) | (k >= 15));
        int jp = (k-15)*((k >= 15)); 

        num_pedestrians = 5;
        times = VectorXf::Zero(num_pedestrians);
        speeds = VectorXf::Zero(num_pedestrians);
        positions = VectorXf::Zero(num_pedestrians);

        lambda_ = (1 / parameters(jT_, jT));
        w = 1.800;
        Ts = 1e-2;
        t_end = 0.00;
        t_start = 0.00;
        pd_v = 0.13;
        pd_sigma = 0.0015600;

        o = parameters(2, jo);
        int j3 = (k-15)*((k >= 15)); 
        stop_probability = parameters(3, jp);
        random_status = VectorXf::Zero(num_pedestrians);
        real_status = VectorXf::Ones(num_pedestrians);
    }

    // Uniform random variable sampler

    VectorXf uniformRV()
    {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        auto uniform = [&](float)
        { return distribution(generator); };
        VectorXf v = VectorXf::NullaryExpr(num_pedestrians, uniform);
        return v;
    }

    // Exponential random variable sampler

    VectorXf exponentialRV(float lam)
    {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::exponential_distribution<float> distribution(lam);
        auto exp = [&](float)
        { return distribution(generator); };
        VectorXf v = VectorXf::NullaryExpr(num_pedestrians, exp);
        return v;
    }

    // Normal random variable sampler

    VectorXf normalRV(float mu, float sigma)
    {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        normal_distribution<float> distribution(mu, sigma);
        auto normal = [&](float)
        { return distribution(generator); };
        VectorXf v = VectorXf::NullaryExpr(num_pedestrians, normal);
        return v;
    }

    // Randomly initialize pedestrians

    void randomInitialize()
    {
        unsigned seed1 = chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator1(seed1);
        std::uniform_real_distribution<float> distribution1(1.00, 100.00);

        float t_sample = distribution1(generator1);
        float t_step = 0;

        times = exponentialRV(lambda_);
        speeds = normalRV(pd_v, pd_sigma);

        real_status = VectorXf::Ones(num_pedestrians);
        random_status = uniformRV();
        positions = VectorXf::Zero(num_pedestrians);

        for (int j = 1; j < times.size(); j++)
        {
            times(j) += times(j - 1);
        }

        while (t_step < t_sample)
        {
            pedestrianPosition(t_step);
            t_step += Ts;
        }

        t_start = t_step;
    }

    // Step pedestrian positions forward one time step

    void updatePedestrian()
    {
        times = exponentialRV(lambda_);
        speeds = normalRV(pd_v, pd_sigma);
        for (int j = 1; j < times.size(); j++)
        {
            times(j) += times(j - 1);
        }
        positions = VectorXf::Zero(num_pedestrians);

        random_status = uniformRV();
    }

    // Reset all pedestrian states to zero

    void resetPedestrian()
    {
        times = VectorXf::Zero(num_pedestrians);
        speeds = VectorXf::Zero(num_pedestrians);
        positions = VectorXf::Zero(num_pedestrians);
        t_end = 0.00;

        random_status = uniformRV();
    }

    // If pedestrian spots vehicle then they stop and avoid it

    void eventPedestrian()
    {
        real_status = (positions.array() > o).select(random_status, 1.00);
        speeds = (real_status.array() > stop_probability).select(speeds, 0.00);
        positions = (real_status.array() > stop_probability).select(positions, o);
    }

    // Get pedestrian positions along crosswalk

    VectorXf pedestrianPosition(float t)
    {
        float t_offset = t - t_end;
        VectorXf mask = (times.array() <= t_offset + t_start).select(VectorXf::Ones(num_pedestrians), 0.00);
        eventPedestrian();
        positions += Ts * (speeds.array() * mask.array()).matrix();
        positions = (positions.array() <= w).select(positions, w);
        if (((positions.array() == w) || (speeds.array() == 0.0)).all())
        {
            updatePedestrian();
            t_end = t;
            t_start = 0;
        }
        return positions;
    }
};

// Simple 1D acceleration based vehicle model

class VehicleDynamics
{
public:

    float b;
    float v_set, P, I;

    VehicleDynamics()
    {
        b = 0e-3;
        v_set = v_cruise;
    }

    // Vehicle cruise controller with acceleration limits

    float ctrl(float t, VectorXf x)
    {
        P = (v_cruise - x(1));
        return clip<float>(0.20*P, -0.030*9.8, 0.030*9.8);
    }

    // Autonomous dynamics function

    VectorXf fs(VectorXf x)
    {
        VectorXf x_dot(2);
        x_dot << x(1), -b * x(1);
        return x_dot;
    }

    // Control affine mapping matrix

    VectorXf gs(VectorXf x)
    {
        VectorXf B(2);
        B << 0, 1;
        return B;
    }

    // Complete system dynamics

    VectorXf f_sys(VectorXf x, float u)
    {
        return fs(x) + gs(x) * u;
    }

    // Closed-loop system dynamics

    VectorXf func(float t, VectorXf x)
    {
        return fs(x) + gs(x) * ctrl(t, x);
    }
    
};

// Safe Probability Class

class SafeProbability
{
    public:
    int N_horizon;
    int N_episode;
    float epsilon = 0.10;
    float Ts, offset_time, d_start, d_crossing, b, y_occ, d_thr, y_pos, y_cross;
    float LfF;
    MatrixXf LgF;
    MatrixXd Q, Ge, Gi;
    VectorXd u0, he, hi, u;
    VehicleDynamics vd;
    PedestrianModel pd;

    SafeProbability(float d_start_, float d_crossing_, float b_, float y_occ_, float d_thr_, float y_pos_, float y_cross_)
    {
        Ts = 5e-2;
        N_horizon = 100;
        N_episode = 5000;
        d_start = d_start_;
        d_crossing = d_crossing_;
        b = b_;
        y_occ = y_occ_;
        d_thr = d_thr_;
        y_pos = y_pos_;
        y_cross = y_cross_;
    }

    // Step simulation for one time-step

    VectorXf stepODE(float t, VectorXf x)
    {
        int n = x.rows();
        MatrixXf K = MatrixXf::Zero(n, 4);
        K.col(0) = Ts*vd.func(t, x);
        K.col(1) = Ts*vd.func(t + Ts/2, x + K.col(0)/2);
        K.col(2) = Ts*vd.func(t + Ts/2, x + K.col(1)/2);
        K.col(3) = Ts*vd.func(t + Ts, x + K.col(2));
        x += (K.col(0) + 2*K.col(1) + 2*K.col(2) + K.col(3))/6;
        return x;
    }

    // Heurisitic vehicle speed modulator

    float velMod(float x)
    {
        float vel;
        float delta = d_thr;
        return v_cruise * ((x < d_crossing - d_start) || (x > d_crossing + delta)) + v_cruise * (1 - (x + d_start - d_crossing)/(d_start - delta)) * ((d_crossing - d_start <= x) && (x < d_crossing - delta)) + 1.00 * ((d_crossing - delta <= x) && (x <= d_crossing + delta));
    }

    // Use vehicle sensors to check and see if a pedestrian is present

    float checkPedestrian(VectorXf x, VectorXf positions)
    {
        float c = ((positions.array() > y_occ) && (positions.array() < y_cross - 1e-2)).any() && (d_crossing - x(0)) <= d_start ? 1.00 : 0.00;
        return c;
    }

    // Evaluate safety violation

    float phi(VectorXf x, VectorXf positions)
    {
        return ((positions.array() > y_pos - b) && (positions.array() < y_pos + b)).any() && (d_crossing - x(0)) <= d_thr && (d_crossing - x(0)) >= -d_thr ? 1.00 : 0.00;
    }

    // Monte-Carlo forward rollout algorithm for estimating safe probability F(t, x)

    float F(float t, VectorXf x)
    {
        int k = round(t/Ts);
        float t_step, c, p;
        VectorXf safeProb = VectorXf::Zero(N_episode);
        VectorXf x_s;
        VectorXf positions;
        for (int i = 0; i < N_episode; i++)
        {
            t_step = t;
            vd.v_set = v_cruise;
            pd.resetPedestrian();
	        pd.randomInitialize();
            x_s = x;
            p = 1.00;
            for (int j = 0; j < N_horizon; j++)
            {
                positions = pd.pedestrianPosition(t_step);
                c = checkPedestrian(x_s, positions);
                vd.v_set = (c == 1) ? velMod(x_s(0)) : v_cruise;
                x_s = stepODE(t_step, x_s);
                t_step += Ts;
                if (phi(x_s, positions) == 0.00)
                {
                    p = 0.0;
                    break;
                }
            }
            safeProb(i, 0) = p;
        }
        return safeProb.mean();
    }
};



int main(int argc, char** argv)
{
    // Convert execution argument to int for parameter case

    k = stoi(argv[1]);

    // Parse parameters file

    CSVData rd("parameters.csv", MatrixXf::Zero(1, 1));

    parameters = rd.readFromCSVfile();

    // Set simulation parameters

    VectorXf x0(2);

    x0 << 0.00, v_cruise;

    int N = 10000;

    float Ts = 1e-2;

    float d_start = 50;

    float d_crossing = 100;

    float b = 0.35;

    float y_occ = parameters(2, 2);

    float d_thr = 1.8/2;

    float y_pos = 0.675;

    float y_cross = 1.8;

    SafeProbability sp(d_start, d_crossing, b, y_occ, d_thr, y_pos, y_cross);

    // Obtain safe probability estimates over meshgrid of system states over a fixed time horizon T

    int Nx = 100;
    int Nv = 100;

    VectorXf x(2);

    ArrayXf x_pos = ArrayXf::LinSpaced(Nx, 0, 200);
    ArrayXf x_vel = ArrayXf::LinSpaced(Nv, 0, 5);

    MatrixXf F_grid = MatrixXf::Zero(Nx, Nv);

    float F_calc;

    for (int i = 0; i < Nx; i++)
    {
        cout << "i: " << i << endl;
        for (int j = 0; j < Nv; j++)
        {
            v_cruise = x_vel(j);
            x << x_pos(i), x_vel(j);
            F_grid(i, j) = sp.F(0, x);
        }
    }

    // Save data

    CSVData sv1("/CaseData/case" + to_string(k+1) + ".csv", F_grid);

    sv1.writeToCSVfile();

    return 0;
}
