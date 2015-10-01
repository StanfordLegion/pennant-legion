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

#include "MyMapper.hh"
#include "InputFile.hh"
#include "Driver.hh"

using namespace std;
using namespace LegionRuntime::HighLevel;

enum TaskID {
    TID_MAIN
};


void registerMappers(
        Machine machine,
        HighLevelRuntime *rt,
        const std::set<Processor> &local_procs)
{
    for (set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
    {
        rt->replace_default_mapper(
                new MyMapper(machine, rt, *it), *it);
    }
}


void mainTask(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, HighLevelRuntime *runtime)
{
    const InputArgs& iargs = HighLevelRuntime::get_input_args();

    // skip over legion args if present
    int i = 1;
    while (i < iargs.argc) {
        string arg(iargs.argv[i], 3);
        if (arg != "-ll" && arg != "-hl" && arg != "-ca" && arg != "-le" && arg != "-dm") break;
        i += 2;
    }
    
    volatile bool debug = false;
    int numpcs = 1;
    const char* filename;
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
      else {
        cerr << "Usage: pennant [legion args] "
             << "[-n <numpcs>] <filename>" << endl;
        exit(1);
      }
    }

    /* spin so debugger can attach... */
    while (debug) {} 
    
    InputFile inp(filename);

    string probname(filename);
    // strip .pnt suffix from filename
    int len = probname.length();
    if (probname.substr(len - 4, 4) == ".pnt")
        probname = probname.substr(0, len - 4);

    Driver drv(&inp, probname, numpcs, ctx, runtime);

    drv.run();

}


int main(int argc, char **argv)
{
    // register main task only; other tasks have already been
    // registered by the classes that own them
    HighLevelRuntime::set_top_level_task_id(TID_MAIN);
    HighLevelRuntime::register_legion_task<mainTask>(
            TID_MAIN, Processor::LOC_PROC, true, false);

    HighLevelRuntime::set_registration_callback(registerMappers);

    return HighLevelRuntime::start(argc, argv);
}

