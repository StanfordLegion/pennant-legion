/*
 * Driver.cc
 *
 *  Created on: Jan 23, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Driver.hh"

#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "InputFile.hh"
#include "Mesh.hh"
#include "Hydro.hh"

using namespace std;


Driver::Driver(
        const InputFile* inp,
        const std::string& pname,
        const int numpcs,
        const bool parallel,
        Context ctx,
        Runtime* runtime)
        : probname(pname) {
    cout << "********************" << endl;
    cout << "Running PENNANT v0.6" << endl;
    cout << "********************" << endl;
    cout << endl;

    cout << "Running Legion on " << numpcs << " piece(s)" << endl;

    cstop = inp->getInt("cstop", 999999);
    tstop = inp->getDouble("tstop", 1.e99);
    if (cstop == 999999 && tstop == 1.e99) {
        cerr << "Must specify either cstop or tstop" << endl;
        exit(1);
    }
    dtmax = inp->getDouble("dtmax", 1.e99);
    dtinit = inp->getDouble("dtinit", 1.e99);
    dtfac = inp->getDouble("dtfac", 1.2);
    dtreport = inp->getInt("dtreport", 10);

    // initialize mesh, hydro
    mesh = new Mesh(inp, numpcs, parallel, ctx, runtime);
    hydro = new Hydro(inp, mesh, ctx, runtime);

}

Driver::~Driver() {

    delete hydro;
    delete mesh;

}

void Driver::run(
        Context ctx,
        Runtime* runtime) {

    time = 0.0;
    cycle = 0;

    // Better timing for Legion
    const TimingLauncher timing_launcher(MEASURE_MICRO_SECONDS);
    std::deque<TimingMeasurement> timing_measurements;
    // First make sure all our setup is done before beginning timing
    runtime->issue_execution_fence(ctx);
    // Get our start time
    Future f_start = runtime->issue_timing_measurement(ctx, timing_launcher);

    // main event loop
    while (cycle < cstop && time < tstop) {

        cycle += 1;

        // get timestep
        calcGlobalDt();

        // begin hydro cycle
        hydro->doCycle(dt);

        time += dt;

        if (cycle == 1 || cycle % dtreport == 0) {

            timing_measurements.push_back(TimingMeasurement());
            TimingMeasurement &measurement = timing_measurements.back();
            measurement.cycle = cycle;
            measurement.f_time = 
              runtime->issue_timing_measurement(ctx, timing_launcher);
            measurement.time = time;
            measurement.dt = dt;
            measurement.msgdt = msgdt;

        } // if cycle...

    } // while cycle...

    // get stopping timestamp
    runtime->issue_execution_fence(ctx);
    Future f_stop = runtime->issue_timing_measurement(ctx, timing_launcher);

    const double tbegin = f_start.get_result<long long>();
    double tlast = tbegin;
    for (std::deque<TimingMeasurement>::const_iterator it = 
          timing_measurements.begin(); it != timing_measurements.end(); it++)
    {
      const double tnext = it->f_time.get_result<long long>();
      const double tdiff = tnext - tlast; 
      cout << scientific << setprecision(5);
      cout << "End cycle " << setw(6) << it->cycle
           << ", time = " << setw(11) << it->time
           << ", dt = " << setw(11) << it->dt
           << ", wall = " << setw(11) << tdiff << endl;
      cout << "dt limiter: " << it->msgdt << endl;
      tlast = tnext;
    }

    const double tend = f_stop.get_result<long long>();
    const double walltime = tend - tbegin;

    // write end message
    cout << endl;
    cout << "Run complete" << endl;
    cout << scientific << setprecision(6);
    cout << "cycle = " << setw(6) << cycle
         << ",         cstop = " << setw(6) << cstop << endl;
    cout << "time  = " << setw(14) << time
         << ", tstop = " << setw(14) << tstop << endl;

    cout << endl;
    cout << "************************************" << endl;
    cout << "hydro cycle run time= " << setw(14) << walltime << endl;
    cout << "************************************" << endl;


    // do final mesh output if we aren't in a parallel mode
    if (!mesh->parallel) {
      hydro->getFinalState();
      mesh->write(probname, cycle, time,
              hydro->zr, hydro->ze, hydro->zp);
    }

}


void Driver::calcGlobalDt() {

    // Save timestep from last cycle
    dtlast = dt;
    msgdtlast = msgdt;

    // Compute timestep for this cycle
    dt = dtmax;
    msgdt = "Global maximum (dtmax)";

    if (cycle == 1) {
        // compare to initial timestep
        if (dtinit < dt) {
            dt = dtinit;
            msgdt = "Initial timestep";
        }
    } else {
        // compare to factor * previous timestep
        double dtrecover = dtfac * dtlast;
        if (dtrecover < dt) {
            dt = dtrecover;
            if (msgdtlast.substr(0, 8) == "Recovery")
                msgdt = msgdtlast;
            else
                msgdt = "Recovery: " + msgdtlast;
        }
    }

    // compare to time-to-end
    if ((tstop - time) < dt) {
        dt = tstop - time;
        msgdt = "Global (tstop - time)";
    }

    // compare to hydro dt
    if (cycle > 1) hydro->getDtHydro(dt, msgdt);

}

