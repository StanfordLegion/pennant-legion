/*
 * PolyGas.cc
 *
 *  Created on: Mar 26, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "PolyGas.hh"

#include "legion.h"

#include "Vec2.hh"
#include "MyLegion.hh"
#include "InputFile.hh"
#include "Hydro.hh"
#include "Mesh.hh"

using namespace std;
using namespace Legion;
using namespace LegionRuntime::Accessor;


namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
    {
      TaskVariantRegistrar registrar(TID_CALCSTATEHALF, "calcstatehalf");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<PolyGas::calcStateHalfTask>(registrar);
    }
    {
      TaskVariantRegistrar registrar(TID_CALCFORCEPGAS, "calcforcepgas");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<PolyGas::calcForceTask>(registrar);
    }
}
}; // namespace


PolyGas::PolyGas(const InputFile* inp, Hydro* h) : hydro(h) {
    gamma = inp->getDouble("gamma", 5. / 3.);
    ssmin = inp->getDouble("ssmin", 0.);
}


void PolyGas::calcStateHalfTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double gamma = args[0];
    const double ssmin = args[1];
    const double dt    = args[2];

    MyAccessor<double> acc_zr =
        get_accessor<double>(regions[0], FID_ZR);
    MyAccessor<double> acc_zvolp =
        get_accessor<double>(regions[0], FID_ZVOLP);
    MyAccessor<double> acc_zvol0 =
        get_accessor<double>(regions[0], FID_ZVOL0);
    MyAccessor<double> acc_ze =
        get_accessor<double>(regions[0], FID_ZE);
    MyAccessor<double> acc_zwrate =
        get_accessor<double>(regions[0], FID_ZWRATE);
    MyAccessor<double> acc_zm =
        get_accessor<double>(regions[0], FID_ZM);
    MyAccessor<double> acc_zp =
        get_accessor<double>(regions[1], FID_ZP);
    MyAccessor<double> acc_zss =
        get_accessor<double>(regions[1], FID_ZSS);

    const double dth = 0.5 * dt;
    const double gm1 = gamma - 1.;
    const double ssmin2 = max(ssmin * ssmin, 1.e-99);
    const IndexSpace& isz = task->regions[0].region.get_index_space();
    for (IndexIterator itrz(runtime,ctx,isz); itrz.has_next(); )
    {
        // compute EOS at beginning of time step
        ptr_t z = itrz.next();
        double r = acc_zr.read(z);
        double e = max(acc_ze.read(z), 0.);
        double p = gm1 * r * e;
        double pre = gm1 * e;
        double per = gm1 * r;
        double csqd = max(ssmin2, pre + per * p / (r * r));
        double ss = sqrt(csqd);

        // now advance pressure to the half-step
        double minv = 1. / acc_zm.read(z);
        double volp = acc_zvolp.read(z);
        double vol0 = acc_zvol0.read(z);
        double wrate = acc_zwrate.read(z);
        double dv = (volp - vol0) * minv;
        double bulk = r * csqd;
        double denom = 1. + 0.5 * per * dv;
        double src = wrate * dth * minv;
        p += (per * src - r * bulk * dv) / denom;
        acc_zp.write(z, p);
        acc_zss.write(z, ss);
    }
}


void PolyGas::calcForceTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    MyAccessor<ptr_t> acc_mapsz =
        get_accessor<ptr_t>(regions[0], FID_MAPSZ);
    MyAccessor<double2> acc_ssurf =
        get_accessor<double2>(regions[0], FID_SSURFP);
    MyAccessor<double> acc_zp =
        get_accessor<double>(regions[1], FID_ZP);
    MyAccessor<double2> acc_sf =
        get_accessor<double2>(regions[2], FID_SFP);

    const IndexSpace& iss = task->regions[0].region.get_index_space();

    for (IndexIterator itrs(runtime,ctx,iss); itrs.has_next(); )
    {
        ptr_t s  = itrs.next();
        ptr_t z  = acc_mapsz.read(s);
        double p = acc_zp.read(z);
        double2 surf = acc_ssurf.read(s);
        double2 sfx = -p * surf;
        acc_sf.write(s, sfx);
    }

}

