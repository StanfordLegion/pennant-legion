/*
 * PolyGas.hh
 *
 *  Created on: Mar 23, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef POLYGAS_HH_
#define POLYGAS_HH_

#include "legion.h"

// forward declarations
class InputFile;
class Hydro;


enum PolyGasTaskID {
    TID_CALCSTATEHALF = 'P' * 100,
    TID_CALCFORCEPGAS
};


class PolyGas {
public:

    // parent hydro object
    Hydro* hydro;

    double gamma;                  // coeff. for ideal gas equation
    double ssmin;                  // minimum sound speed for gas

    PolyGas(const InputFile* inp, Hydro* h);
    ~PolyGas();

    static void calcStateHalfTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);
    
    // OpenMP variant
    static void calcStateHalfOMPTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // GPU variant
    static void calcStateHalfGPUTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void calcForceTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // OpenMP variant
    static void calcForceOMPTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // GPU variant
    static void calcForceGPUTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

};  // class PolyGas


#endif /* POLYGAS_HH_ */
