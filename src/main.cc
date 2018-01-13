/*
 * main.cc
 *
 *  Created on: Jan 23, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include <cstdlib>
#include <string>
#include <iostream>

#include "legion.h"

#include "PennantMapper.hh"
#include "InputFile.hh"
#include "Driver.hh"

using namespace std;
using namespace Legion;

enum TaskID {
    TID_MAIN
};


void registerMappers(
        Machine machine,
        Runtime *rt,
        const std::set<Processor> &local_procs)
{
    for (set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
    {
        rt->replace_default_mapper(
                new PennantMapper(machine, rt, *it), *it);
    }
}


void mainTask(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, Runtime*runtime)
{
    const InputArgs& iargs = Runtime::get_input_args();

    // skip over legion args if present
    int i = 1;
    while (i < iargs.argc) {
        string arg(iargs.argv[i], 3);
        if (arg != "-ll" && arg != "-lg" && arg != "-ca" && arg != "-le" && arg != "-dm") break;
        i += 2;
    }
    
    volatile bool debug = false;
    int numpcs = 1;
    bool sequential = false;
    const char* filename = NULL;
    bool warn = true;
    while (i < iargs.argc) { 
      if (iargs.argv[i] == string("-f")) { 
        filename = iargs.argv[i+1];
        i += 2;
      }
      else if (iargs.argv[i] == string("-d")) {
        debug = true;
        i++;
      }
      else if (iargs.argv[i] == string("-n")) {
        numpcs = atoi(iargs.argv[i + 1]);
        i += 2;
      }
      else if (iargs.argv[i] == string("-s")) {
        sequential = true;
        i++;
      }
      else {
        if (warn) {
          LEGION_PRINT_ONCE(runtime, ctx, stderr, "Usage: pennant [legion args] "
                                                   "[-n <numpcs>] <filename>\n");
          warn = false;
        }
        i++;
      }
    }

    if (filename == NULL) {
      LEGION_PRINT_ONCE(runtime, ctx, stderr, "No Pennant input file specified. Exitting...\n");
      return; 
    }

    /* spin so debugger can attach... */
    while (debug) {} 
    
    InputFile inp(filename);

    string probname(filename);
    // strip .pnt suffix from filename
    int len = probname.length();
    if (probname.substr(len - 4, 4) == ".pnt")
        probname = probname.substr(0, len - 4);

    Driver drv(&inp, probname, numpcs, !sequential, ctx, runtime);

    drv.run();

}


int main(int argc, char **argv)
{
    // register main task only; other tasks have already been
    // registered by the classes that own them
    Runtime::set_top_level_task_id(TID_MAIN);
    TaskVariantRegistrar registrar(TID_MAIN, "main");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner(true);
    registrar.set_replicable(true);
    Runtime::preregister_task_variant<mainTask>(registrar, "main");

    Runtime::add_registration_callback(registerMappers);

    return Runtime::start(argc, argv);
}

