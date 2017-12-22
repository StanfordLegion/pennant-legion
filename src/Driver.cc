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

void Driver::run() {

    time = 0.0;
    cycle = 0;

    double tbegin, tlast;
    // get starting timestamp
    struct timeval sbegin;
    gettimeofday(&sbegin, NULL);
    tbegin = sbegin.tv_sec + sbegin.tv_usec * 1.e-6;
    tlast = tbegin;

    // main event loop
    while (cycle < cstop && time < tstop) {

        cycle += 1;

        // get timestep
        calcGlobalDt();

        // begin hydro cycle
        hydro->doCycle(dt);

        time += dt;

        if (cycle == 1 || cycle % dtreport == 0) {
            struct timeval scurr;
            gettimeofday(&scurr, NULL);
            double tcurr = scurr.tv_sec + scurr.tv_usec * 1.e-6;
            double tdiff = tcurr - tlast;

            cout << scientific << setprecision(5);
            cout << "End cycle " << setw(6) << cycle
                 << ", time = " << setw(11) << time
                 << ", dt = " << setw(11) << dt
                 << ", wall = " << setw(11) << tdiff << endl;
            cout << "dt limiter: " << msgdt << endl;

            tlast = tcurr;
        } // if cycle...

    } // while cycle...

    // get stopping timestamp
    struct timeval send;
    gettimeofday(&send, NULL);
    double tend = send.tv_sec + send.tv_usec * 1.e-6;
    double runtime = tend - tbegin;

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
    cout << "hydro cycle run time= " << setw(14) << runtime << endl;
    cout << "************************************" << endl;


    // do final mesh output
    hydro->getFinalState();
    mesh->write(probname, cycle, time,
            hydro->zr, hydro->ze, hydro->zp);

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

