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
using namespace ResilientLegion;

namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
    {
      TaskVariantRegistrar registrar(TID_CALCGLOBALDT, "CPU calc global dt");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<double,Driver::calcGlobalDtTask>(registrar, "calc global dt");
    }
    {
      TaskVariantRegistrar registrar(TID_UPDATETIME, "CPU update time");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<double,Driver::updateTimeTask>(registrar, "update time");
    }
    {
      TaskVariantRegistrar registrar(TID_UPDATECYCLE, "CPU calc global dt");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<int,Driver::updateCycleTask>(registrar, "update cycle");
    }
    {
      TaskVariantRegistrar registrar(TID_TESTNOTDONE, "CPU test not done");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<bool,Driver::testNotDoneTask>(registrar, "test not done");
    }
    {
      TaskVariantRegistrar registrar(TID_REPORTMEASUREMENT, "CPU report measurement");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Driver::reportMeasurementTask>(registrar, "report measurement");
    }
}
}; // namespace

Driver::Driver(
        const InputFile* inp,
        const std::string& pname,
        const int numpcs,
        Context c,
        Runtime* rt)
        : probname(pname), ctx(c), runtime(rt) {
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
    mesh = new Mesh(inp, numpcs, ctx, runtime);
    hydro = new Hydro(inp, mesh, ctx, runtime);

}

Driver::~Driver() {

    delete hydro;
    delete mesh;

}

void Driver::run(void) {

    Predicate p_not_done = Predicate::TRUE_PRED;
    const double time_init = 0.0;
    Future f_time = Future::from_value(time_init);
    const int cycle_init = 0;
    Future f_cycle = Future::from_value(cycle_init);
    // Need to give these dummy values so we can trace consistently
    Future f_dt = Future::from_value(runtime, 0.0);
    Future f_cdt = Future::from_value(runtime, 0.0);
    Future f_prev_report;
    // Create a trace ID for all of Pennant to use
    const TraceID trace_id = 
      runtime->generate_library_trace_ids("pennant", 1/*one ID*/);

    // Better timing for Legion
    TimingLauncher timing_launcher(MEASURE_MICRO_SECONDS);
    //std::deque<TimingMeasurement> timing_measurements;
    // First make sure all our setup is done before beginning timing
    runtime->issue_execution_fence(ctx);
    // Get our start time
    Future f_start = runtime->issue_timing_measurement(ctx, timing_launcher);
    Future f_prev_measurement = f_start;

    // main event loop
    for (int cycle = 0; cycle < cstop; cycle++) {
        runtime->checkpoint(ctx);

        runtime->begin_trace(ctx, trace_id);
        // get timestep
        f_dt = calcGlobalDt(f_dt, f_cdt, f_time, cycle, p_not_done);

        // begin hydro cycle
        f_cdt = hydro->doCycle(f_dt, cycle, p_not_done);

        f_time = update_time(f_time, f_dt, p_not_done);

        f_cycle = update_cycle(f_cycle, p_not_done);

#ifdef ENABLE_MAX_CYCLE_PREDICATION
        Future f_not_done = test_not_done(f_time, p_not_done); 

        p_not_done = runtime->create_predicate(ctx, f_not_done);
#endif
        runtime->end_trace(ctx, trace_id);

        if ((cycle == 0) || (((cycle+1) % dtreport) == 0)) {
            timing_launcher.preconditions.clear();
            // Measure after f_cdt is ready which is when the cycle is complete
            timing_launcher.add_precondition(f_cdt);
            Future f_measurement = 
              runtime->issue_timing_measurement(ctx, timing_launcher);
            f_prev_report = report_measurement(f_measurement, f_prev_measurement, 
                cycle, f_prev_report, f_time, f_cycle, f_dt, p_not_done);
            f_prev_measurement = f_measurement;
        } // if cycle...

    } // for cycle...

    // get stopping timestamp
    timing_launcher.preconditions.clear();
    // Measure after f_cdt is ready which is when the cycle is complete
    timing_launcher.add_precondition(f_cdt);
    Future f_stop = runtime->issue_timing_measurement(ctx, timing_launcher);

    const double tbegin = f_start.get_result<long long>(true/*silence warnings*/);
    const double tend = f_stop.get_result<long long>(true/*silence warnings*/);
    const double walltime = tend - tbegin;
    // Make sure that all the previous measurements are done being reported
    // before we write out any of the final information
    f_prev_report.get_void_result(true/*silence warnings*/);

    // write end message
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "\nRun complete\n");
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "cycle = %6d,        cstop = %8d\n", f_cycle.get_result<int>(), cstop);
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "time = %14.6g, tstop = %8.6g\n\n", f_time.get_result<double>(), tstop);

    LEGION_PRINT_ONCE(runtime, ctx, stdout, "************************************\n");
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "hydro cycle run time= %14.8g us\n", walltime);
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "************************************\n");

    // Write out any output from running this
    // Note this is inherently not scalable in its current implementation so you can skip
    // it if it is causing you problems by trying to suck all the data to one node to 
    // write it out to individual files.
    mesh->write(probname, f_cycle, f_time);
}


Future Driver::calcGlobalDt(
                    Future f_dt, 
                    Future f_cdt, 
                    Future f_time,
                    const int cycle,
                    Predicate pred) {
  GlobalDtArgs args;
  args.dtinit = dtinit;
  args.dtmax = dtmax;
  args.dtfac = dtfac;
  args.tstop = tstop;
  args.cycle = cycle;
  TaskLauncher launcher(TID_CALCGLOBALDT, TaskArgument(&args, sizeof(args)), pred);
  launcher.set_predicate_false_future(f_dt);
  launcher.add_future(f_time);
  // These won't be read on the first cycle but add them anyway so that the
  // trace is valid
  launcher.add_future(f_dt);
  launcher.add_future(f_cdt);
  return runtime->execute_task(ctx, launcher);
}


double Driver::calcGlobalDtTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
  const GlobalDtArgs *args = reinterpret_cast<const GlobalDtArgs*>(task->args);

  double dt = args->dtmax;
  if (args->cycle == 0) {
    // compare to initial timestep
    if (args->dtinit < dt)
      dt = args->dtinit;
  } else {
    const double dtlast = task->futures[1].get_result<double>();
    // compare to factor * previous timestep
    const double dtrecover = args->dtfac * dtlast;
    if (dtrecover < dt)
      dt = dtrecover;
  }

  // compare to time-to-end
  const double time = task->futures[0].get_result<double>();
  if ((args->tstop - time) < dt)
    dt = args->tstop - time;

  // compare to hydro dt
  if (args->cycle > 0) {
    const double dtrec = task->futures[2].get_result<double>();
    if (dtrec < dt)
      dt = dtrec;
  }
  return dt;
}


Future Driver::update_time(
                  Future f_time, 
                  Future f_dt, 
                  Predicate pred) {
  TaskLauncher launcher(TID_UPDATETIME, TaskArgument(), pred);
  launcher.set_predicate_false_future(f_time);
  launcher.add_future(f_time);
  launcher.add_future(f_dt);
  return runtime->execute_task(ctx, launcher);
}

double Driver::updateTimeTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
  const double time = task->futures[0].get_result<double>();
  const double dt = task->futures[1].get_result<double>();
  return time + dt;
}


Future Driver::update_cycle(
                  Future f_cycle,
                  Predicate pred) {
  TaskLauncher launcher(TID_UPDATECYCLE, TaskArgument(), pred);
  launcher.set_predicate_false_future(f_cycle);
  launcher.add_future(f_cycle);
  return runtime->execute_task(ctx, launcher);
}

int Driver::updateCycleTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
  const int cycle = task->futures[0].get_result<int>();
  return cycle + 1;
}


Future Driver::test_not_done(
                  Future f_time,
                  Predicate pred) {
  TaskLauncher launcher(TID_TESTNOTDONE, TaskArgument(&tstop, sizeof(tstop)), pred);
  const bool false_val = false;
  launcher.set_predicate_false_result(TaskArgument(&false_val, sizeof(false_val)));
  launcher.add_future(f_time);
  return runtime->execute_task(ctx, launcher);
}


bool Driver::testNotDoneTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
  const double tstop = *reinterpret_cast<const double*>(task->args);
  const double time = task->futures[0].get_result<double>();
  return (time < tstop);
}

Future Driver::report_measurement(
                  Future f_measurement,
                  Future f_prev_measurement,
                  const int cycle,
                  Future f_prev_report,
                  Future f_time,
                  Future f_cycle,
                  Future f_dt,
                  Predicate pred) {
  TaskLauncher launcher(TID_REPORTMEASUREMENT, TaskArgument(), pred);
  launcher.set_predicate_false_future(f_prev_report);
  launcher.add_future(f_measurement);
  launcher.add_future(f_prev_measurement);
  launcher.add_future(f_time);
  launcher.add_future(f_cycle);
  launcher.add_future(f_dt);
  // This part guarantees that measurements are printed in order
  if (cycle > 0)
    launcher.add_future(f_prev_report);
  return runtime->execute_task(ctx, launcher);
}


void Driver::reportMeasurementTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
  const long long measurement = task->futures[0].get_result<long long>();
  const long long previous = task->futures[1].get_result<long long>();
  const double time = task->futures[2].get_result<double>();
  const int cycle = task->futures[3].get_result<int>();
  const double dt = task->futures[4].get_result<double>();
  const long long tdiff = measurement - previous; 
  fprintf(stdout, "End cycle %6d, time = %11.5g, dt = %11.5g, wall = %11lld us\n", 
          cycle, time, dt, tdiff);
  fflush(stdout);
}


