#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <chrono>
#include <random>
#include <functional>
#include <libInterpolate/Interpolators/_2D/BicubicInterpolator.hpp>
#include <libInterpolate/Interpolators/_2D/BilinearInterpolator.hpp> 
#include "eiquadprog.hpp"

using namespace std;
using namespace Eigen;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

// Clip function

template <typename T>
T clip(const T& n, const T& lower, const T& upper)
{
    return std::max(lower, std::min(n, upper));
}

MatrixXf parameters;

int k = 0;

// Create a meshgrid -> identical to numpy.meshgrid

MatrixXf meshgrid(VectorXf x, VectorXf y)
{

    int n = x.size();
    int m = y.size();
    MatrixXf xx = x.transpose().replicate(m, 1);
    MatrixXf yy = y.replicate(1, n);

    MatrixXf M(2*m, n);

    M << xx, yy;

    return M;

}

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

// Libinterpolator wrapper function

class Interpolator
{
    public:
    _2D::BicubicInterpolator<float> interp;
    // _2D::BilinearInterpolator<float> interp;

    Interpolator()
    {
        CSVData csv("/home/schmidd/Documents/IFAC-Latest/IFAC-AV-Occlusion-main/VehicleSimulation/CaseData/case" + to_string(k+1) + "_filt.csv", MatrixXf::Zero(1, 1));
        MatrixXf F = csv.readFromCSVfile();
        int nx = F.cols(); int nv = F.rows();
        MatrixXf M = meshgrid(VectorXf::LinSpaced(nx, 0, 200), VectorXf::LinSpaced(nv, 0, 5));
        MatrixXf xx = M.block(0, 0, nv, nx);
        MatrixXf vv = M.block(nv, 0, nv, nx);
        interp.setData( xx, vv, F );
    }

    float getInterp(float x, float v)
    {
        return interp(x, v);
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

// Safe Controller Class

class SafeController
{
    public:
    int N_horizon;
    int N_episode;
    float epsilon = 0.200;
    float Ts, offset_time, d_start, d_crossing, b, y_occ, d_thr, y_pos, y_cross;
    float LfF;
    MatrixXf LgF;
    MatrixXd Q, Ge, Gi;
    VectorXd u0, he, hi, u;
    VehicleDynamics vd;
    PedestrianModel pd;
    Interpolator interp;

    SafeController(float d_start_, float d_crossing_, float b_, float y_occ_, float d_thr_, float y_pos_, float y_cross_)
    {
        Ts = 5e-2;
        N_horizon = 400;
        N_episode = 1000;
        d_start = d_start_;
        d_crossing = d_crossing_;
        b = b_;
        y_occ = y_occ_;
        d_thr = d_thr_;
        y_pos = y_pos_;
        y_cross = y_cross_;
    }

    // QP solver wrapper function

    VectorXf solveQP(VectorXf u0_, MatrixXf Gi_, VectorXf hi_)
    {
        u0 = -1*u0_.cast <double> ();
        Gi = (Gi_.transpose()).cast <double> ();
        hi = hi_.cast <double> ();
        Q = MatrixXd::Identity(u0.rows(), u0.rows());
        Ge.resize(Q.rows(), 0);
        he.resize(0);
        solve_quadprog(Q, u0, Ge, he, Gi, hi, u);
        VectorXf u_f = u.cast <float> ();
        return u_f;
    }

    // Get safe probability from bicubic interpolator function and apply boundary conditions

    float F(float t, VectorXf x)
    {   
        float F_calc = ( (x(0) >= 0) & (x(0) <= 200) & (x(1) >= 0) & (x(1) <= 5) ) ? interp.getInterp(x(0), x(1)) : 1.00;
        return (F_calc <= 1) ? F_calc : 1.00;
    }

    // Compute safe probability gradient

    VectorXf gradF(float t, VectorXf x)
    {
        int n = x.size();

        VectorXf dF = VectorXf::Zero(n);

        MatrixXf eye = MatrixXf::Identity(n, n);
        
        eye(0, 0) = 1.0; eye(1, 1) = 0.01;

        for (int j = 0; j < n; j++)
        {
            dF(j, 0) = ( F(t, x + eye.col(j)) - F(t, x - eye.col(j)) ) / (2*eye(j, j));
        }

        return dF;
    }

    // Compute lie-derivatives of F along fs and gs

    void calcLF(float t, VectorXf x)
    {
        VectorXf dF = gradF(t, x);
        LfF = vd.fs(x).dot(dF);
        LgF = dF.transpose() * vd.gs(x);
    }

    // Compute safe control action by solving QP

    VectorXf safeCtrl(float t, VectorXf x, VectorXf uN)
    {
        calcLF(t, x);
        int m = uN.rows();
        MatrixXf Gi = LgF;
        VectorXf hi(1); hi << LfF + 1.00*(F(t, x) - (1 - epsilon));
        VectorXf u = solveQP(uN, Gi, hi);
        return u;
    }
};

// RK4 ODE solver

class SolveODE
{
    public:
    float Ts, offset_time, d_start, d_crossing, b, y_occ, d_thr, y_pos, y_cross;
    int N, Nh;
    VectorXf x0, u0;
    MatrixXf Q, R, X_ref, U_ref;
    VehicleDynamics vd;
    PedestrianModel pd;

    SolveODE(float Ts_, int N_, float d_start_, float d_crossing_, float b_, float y_occ_, float d_thr_, float y_pos_, float y_cross_, VectorXf x0_)
    {
        Ts = Ts_;
        N = N_;
        d_start = d_start_;
        d_crossing = d_crossing_;
        b = b_;
        y_occ = y_occ_;
        d_thr = d_thr_;
        y_pos = y_pos_;
        y_cross = y_cross_;
        x0 = x0_;
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

    // Solve ODE

    void solve()
    {
        int n = x0.rows();
        VectorXf u_(1);
        VectorXf u_safe;
        MatrixXf X = MatrixXf::Zero(n, N);
        MatrixXf K = MatrixXf::Zero(n, 4);
        VectorXf positions;
        MatrixXf positions_data = MatrixXf::Zero(5, N);
        MatrixXf safeProb = MatrixXf::Zero(N, 1);
        MatrixXf violations = MatrixXf::Zero(N, 1);
        SafeController sp(d_start, d_crossing, b, y_occ, d_thr, y_pos, y_cross);
        X.col(0) = x0;
        safeProb(0, 0) = sp.F(0, x0);
        float t;
        float c;
        vd.v_set = v_cruise;
        pd.updatePedestrian();
        for (int i = 0; i < N-1; i++)
        {
            t = i*Ts;

            positions = pd.pedestrianPosition(t);

            positions_data.col(i) = positions;

            c = checkPedestrian(X.col(i), positions);

            vd.v_set = (c == 1) ? velMod(X(0, i)) : v_cruise;

            u_ << vd.ctrl(t, X.col(i));

            u_safe = sp.safeCtrl(t, X.col(i), u_);


            K.col(0) = Ts*(vd.fs(X.col(i)) + vd.gs(X.col(i)) * u_safe);
            K.col(1) = Ts*(vd.fs(X.col(i) + K.col(0)/2) + vd.gs(X.col(i) + K.col(0)/2) * u_safe);
            K.col(2) = Ts*(vd.fs(X.col(i) + K.col(1)/2) + vd.gs(X.col(i) + K.col(1)/2) * u_safe);
            K.col(3) = Ts*(vd.fs(X.col(i) + K.col(2)) + vd.gs(X.col(i) + K.col(2)) * u_safe);
            X.col(i+1) = X.col(i) + (K.col(0) + 2*K.col(1) + 2*K.col(2) + K.col(3))/6;
            safeProb(i+1, 0) = sp.F(t+Ts, X.col(i+1));
            violations(i+1, 0) = phi(X.col(i+1), positions);
        }

    // Save solved ODE data

    positions_data.col(N-1) = pd.pedestrianPosition(t + Ts);

    CSVData sv1("/SimData/pedestrianData_" + to_string(k+1) + ".csv", positions_data.transpose());

    sv1.writeToCSVfile();

    CSVData sv2("/SimData/safeProbData_" + to_string(k+1) + ".csv", safeProb);

    sv2.writeToCSVfile();

    CSVData sv3("/SimData/stateData_" + to_string(k+1) + ".csv", X.transpose());

    sv3.writeToCSVfile();

    CSVData sv4("/SimData/violations_" + to_string(k+1) + ".csv", violations);

    sv4.writeToCSVfile();

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

    v_cruise = 4.0;

    VectorXf x0(2);

    x0 << 0.00, v_cruise;

    int N = 25000;

    float Ts = 1e-2;

    float d_start = 50;

    float d_crossing = 100;

    float b = 0.35;

    int jo = (k-10)*(k >= 10 & k < 15) + 2*(k < 10 | k >= 15);

    float y_occ = parameters(2, jo);

    float d_thr = 1.8/2;

    float y_pos = 0.675;

    float y_cross = 1.8;

    cout << "y_occ: " << y_occ << endl; 

    cout << "v_cruise: " << v_cruise << endl;

    // Solve ODE

    SafeController sp(d_start, d_crossing, b, y_occ, d_thr, y_pos, y_cross);

    SolveODE ode (Ts, N, d_start, d_crossing, b, y_occ, d_thr, y_pos, y_cross, x0);

    ode.solve();

    return 0;
}
