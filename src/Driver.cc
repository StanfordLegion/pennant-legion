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
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "********************\n");
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "Running PENNANT v0.6\n");
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "********************\n\n");

    LEGION_PRINT_ONCE(runtime, ctx, stdout, "Running Legion on %d piece(s)", numpcs);

    cstop = inp->getInt("cstop", 999999);
    tstop = inp->getDouble("tstop", 1.e99);
    if (cstop == 999999 && tstop == 1.e99) {
        LEGION_PRINT_ONCE(runtime, ctx, stderr, "Must specify either cstop or tstop\n");
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
    TimingLauncher timing_launcher(MEASURE_MICRO_SECONDS);
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
        hydro->doCycle(dt, cycle);

        time += dt;

        if (cycle == 1 || cycle % dtreport == 0) {

            timing_launcher.preconditions.clear();
            timing_launcher.add_precondition(hydro->f_cdt);
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
      LEGION_PRINT_ONCE(runtime, ctx, stdout, "End cycle %6d, time = %11.5g"
          ", dt = %11.5g, wall = %11.5g us\n", it->cycle, it->time, it->dt, tdiff);
      LEGION_PRINT_ONCE(runtime, ctx, stdout, "dt limiter: %s\n", it->msgdt.c_str());
      tlast = tnext;
    }

    const double tend = f_stop.get_result<long long>();
    const double walltime = tend - tbegin;

    // write end message
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "\nRun complete\n");
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "cycle = %6d,        cstop = %6d\n", cycle, cstop);
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "time = %14.6g, tstop = %14.6g\n\n", time, tstop);

    LEGION_PRINT_ONCE(runtime, ctx, stdout, "************************************\n");
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "hydro cycle run time= %14.8g us\n", walltime);
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "************************************\n");


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

