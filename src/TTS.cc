/*
 * TTS.cc
 *
 *  Created on: Feb 2, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "TTS.hh"

#include "legion.h"

#include "MyLegion.hh"
#include "Vec2.hh"
#include "InputFile.hh"
#include "Mesh.hh"
#include "Hydro.hh"

using namespace std;
using namespace Legion;
using namespace LegionRuntime::Accessor;


namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
    TaskVariantRegistrar registrar(TID_CALCFORCETTS, "calcforcetts");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<TTS::calcForceTask>(registrar);
}
}; // namespace


TTS::TTS(const InputFile* inp, Hydro* h) : hydro(h) {
    alfa = inp->getDouble("alfa", 0.5);
    ssmin = inp->getDouble("ssmin", 0.);

}


TTS::~TTS() {}


void TTS::calcForceTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double alfa  = args[0];
    const double ssmin = args[1];

    MyAccessor<ptr_t> acc_mapsz =
        get_accessor<ptr_t>(regions[0], FID_MAPSZ);
    MyAccessor<double> acc_sarea =
        get_accessor<double>(regions[0], FID_SAREAP);
    MyAccessor<double> acc_smf =
        get_accessor<double>(regions[0], FID_SMF);
    MyAccessor<double2> acc_ssurf =
        get_accessor<double2>(regions[0], FID_SSURFP);
    MyAccessor<double> acc_zarea =
        get_accessor<double>(regions[1], FID_ZAREAP);
    MyAccessor<double> acc_zr =
        get_accessor<double>(regions[1], FID_ZRP);
    MyAccessor<double> acc_zss =
        get_accessor<double>(regions[1], FID_ZSS);
    MyAccessor<double2> acc_sf =
        get_accessor<double2>(regions[2], FID_SFT);

    //  Side density:
    //    srho = sm/sv = zr (sm/zm) / (sv/zv)
    //  Side pressure:
    //    sp   = zp + alfa dpdr (srho-zr)
    //         = zp + sdp
    //  Side delta pressure:
    //    sdp  = alfa dpdr (srho-zr)
    //         = alfa c**2 (srho-zr)
    //
    //    Notes: smf stores (sm/zm)
    //           svfac stores (sv/zv)

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (IndexIterator itrs(runtime,ctx,iss); itrs.has_next(); )
    {
        ptr_t s  = itrs.next();
        ptr_t z  = acc_mapsz.read(s);
        double sarea = acc_sarea.read(s);
        double zarea = acc_zarea.read(z);
        double vfacinv = zarea / sarea;
        double r = acc_zr.read(z);
        double mf = acc_smf.read(s);
        double srho = r * mf * vfacinv;
        double ss = acc_zss.read(z);
        double sstmp = max(ss, ssmin);
        sstmp = alfa * sstmp * sstmp;
        double sdp = sstmp * (srho - r);
        double2 surf = acc_ssurf.read(s);
        double2 sqq = -sdp * surf;
        acc_sf.write(s, sqq);
    }

}

