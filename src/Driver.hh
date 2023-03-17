/*
 * Driver.hh
 *
 *  Created on: Jan 23, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef DRIVER_HH_
#define DRIVER_HH_

#include <string>

#include "resilience.h"
#define Legion ResilientLegion

enum DriverTaskID {
    TID_CALCGLOBALDT = 'D' * 100,
    TID_UPDATETIME,
    TID_UPDATECYCLE,
    TID_TESTNOTDONE,
    TID_REPORTMEASUREMENT
};

// forward declarations
class InputFile;
class Mesh;
class Hydro;


class Driver {
public:
    struct __attribute__ ((packed)) GlobalDtArgs {
    public:
      double dtinit;
      double dtmax;
      double dtfac;
      double tstop;
      int cycle;
    };
    struct TimingMeasurement {
    public:
      int cycle;
      Legion::Future f_time;
      double time;
      double dt;
      std::string msgdt;
    };
public:

    // children of this object
    Mesh *mesh;
    Hydro *hydro;

    std::string probname;          // problem name
    //double time;                   // simulation time
    //int cycle;                     // simulation cycle number
    double tstop;                  // simulation stop time
    int cstop;                     // simulation stop cycle
    double dtmax;                  // maximum timestep size
    double dtinit;                 // initial timestep size
    double dtfac;                  // factor limiting timestep growth
    int dtreport;                  // frequency for timestep reports
    //double dt;                     // current timestep
    //double dtlast;                 // previous timestep
    std::string msgdt;             // dt limiter message
    std::string msgdtlast;         // previous dt limiter message
    Legion::Context ctx;
    Legion::Runtime* runtime;

    Driver(
            const InputFile* inp,
            const std::string& pname,
            const int numpcs,
            Legion::Context ctx,
            Legion::Runtime* runtime);
    ~Driver();

    void run(void);
    Legion::Future calcGlobalDt(Legion::Future f_dt,
                                Legion::Future f_cdt,
                                Legion::Future f_time,
                                const int cycle,
                                Legion::Predicate pred);

    Legion::Future update_time(Legion::Future f_time,
                               Legion::Future f_dt,
                               Legion::Predicate pred);

    Legion::Future update_cycle(Legion::Future f_cycle,
                                Legion::Predicate pred);

    Legion::Future test_not_done(Legion::Future f_time,
                                 Legion::Predicate pred);

    Legion::Future report_measurement(Legion::Future f_measurement,
                                      Legion::Future f_prev_measurement,
                                      const int cycle,
                                      Legion::Future f_prev_report,
                                      Legion::Future f_time,
                                      Legion::Future f_cycle,
                                      Legion::Future f_dt,
                                      Legion::Predicate pred);

    static double calcGlobalDtTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static double updateTimeTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static int updateCycleTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static bool testNotDoneTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void reportMeasurementTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

};  // class Driver


#endif /* DRIVER_HH_ */
