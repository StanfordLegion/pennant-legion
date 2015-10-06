/*
 * Hydro.hh
 *
 *  Created on: Dec 22, 2011
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef HYDRO_HH_
#define HYDRO_HH_

#include <string>
#include <vector>

#include "legion.h"

#include "Mesh.hh"
#include "Vec2.hh"

// forward declarations
class InputFile;
class Mesh;
class PolyGas;
class TTS;
class QCS;
class HydroBC;


enum HydroTaskID {
    TID_ADVPOSHALF = 'H' * 100,
    TID_CALCRHO,
    TID_CALCCRNRMASS,
    TID_SUMCRNRFORCE,
    TID_CALCACCEL,
    TID_ADVPOSFULL,
    TID_CALCWORK,
    TID_CALCWORKRATE,
    TID_CALCENERGY,
    TID_CALCDT
};


class Hydro {
public:

    // associated mesh object
    Mesh* mesh;

    // children of this object
    PolyGas* pgas;
    TTS* tts;
    QCS* qcs;
    std::vector<HydroBC*> bcs;

    LegionRuntime::HighLevel::Context ctx;
    LegionRuntime::HighLevel::HighLevelRuntime* runtime;

    double cfl;                 // Courant number, limits timestep
    double cflv;                // volume change limit for timestep
    double rinit;               // initial density for main mesh
    double einit;               // initial energy for main mesh
    double rinitsub;            // initial density in subregion
    double einitsub;            // initial energy in subregion
    double uinitradial;         // initial velocity in radial direction
    std::vector<double> bcx;    // x values of x-plane fixed boundaries
    std::vector<double> bcy;    // y values of y-plane fixed boundaries

    LegionRuntime::HighLevel::FutureMap fmapcdt;
    double dtrec;               // maximum timestep for hydro
    std::string msgdtrec;       // message:  reason for dtrec

    double2* pu;       // point velocity

    double* zm;        // zone mass
    double* zr;        // zone density
    double* ze;        // zone specific internal energy
                       // (energy per unit mass)
    double* zetot;     // zone total internal energy
    double* zwrate;    // zone work rate
    double* zp;        // zone pressure

    Hydro(
            const InputFile* inp,
            Mesh* m,
            LegionRuntime::HighLevel::Context ctxa,
            LegionRuntime::HighLevel::HighLevelRuntime* runtimea);
    ~Hydro();

    void init();

    void initRadialVel(
            const double vel,
            const int pfirst,
            const int plast);

    void doCycle(const double dt);

    void getFinalState();

    static void advPosHalfTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

    static void calcRhoTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

    static void calcCrnrMassTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

    static void sumCrnrForceTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

    static void calcAccelTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

    static void advPosFullTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

    static void calcWorkTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

    static void calcWorkRateTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

    static void calcEnergyTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

    static double calcDtTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

    void getDtHydro(
            double& dtnew,
            std::string& msgdtnew);

    void resetDtHydro();

}; // class Hydro



#endif /* HYDRO_HH_ */
