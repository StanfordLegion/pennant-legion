/*
 * TTS.hh
 *
 *  Created on: Feb 2, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef TTS_HH_
#define TTS_HH_

#include "legion.h"

// forward declarations
class InputFile;
class Hydro;


enum TTSTaskID {
    TID_CALCFORCETTS = 'T' * 100
};


class TTS {
public:

    // parent hydro object
    Hydro* hydro;

    double alfa;                   // alpha coefficient for TTS model
    double ssmin;                  // minimum sound speed

    TTS(const InputFile* inp, Hydro* h);
    ~TTS();

    static void calcForceTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

}; // class TTS


#endif /* TTS_HH_ */
