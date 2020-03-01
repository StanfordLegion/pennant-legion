/*
 * Hydro.cc
 *
 *  Created on: Dec 22, 2011
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Hydro.hh"

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include "legion.h"

#include "MyLegion.hh"
#include "Memory.hh"
#include "InputFile.hh"
#include "Mesh.hh"
#include "PolyGas.hh"
#include "TTS.hh"
#include "QCS.hh"
#include "HydroBC.hh"
#include "PennantMapper.hh"

using namespace std;
using namespace Memory;
using namespace Legion;

namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
    {
      TaskVariantRegistrar registrar(TID_ADVPOSHALF, "CPU advposhalf");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::advPosHalfTask>(registrar, "advposhalf");
    }
    {
      TaskVariantRegistrar registrar(TID_ADVPOSHALF, "OMP advposhalf");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::advPosHalfOMPTask>(registrar, "advposhalf");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCRHO, "CPU calcrho");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcRhoTask>(registrar, "calcrho");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCRHO, "OMP calcrho");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcRhoOMPTask>(registrar, "calcrho");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCCRNRMASS, "CPU calccrnrmass");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcCrnrMassTask>(registrar, "calccrnrmass");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCCRNRMASS, "OMP calccrnrmass");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcCrnrMassOMPTask>(registrar, "calccrnrmass");
    }
    {
      TaskVariantRegistrar registrar(TID_SUMCRNRFORCE, "CPU sumcrnrforce");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::sumCrnrForceTask>(registrar, "sumcrnrforce");
    }
    {
      TaskVariantRegistrar registrar(TID_SUMCRNRFORCE, "OMP sumcrnrforce");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::sumCrnrForceOMPTask>(registrar, "sumcrnrforce");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCACCEL, "CPU calcaccel");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcAccelTask>(registrar, "calcaccel"); 
    }
    {
      TaskVariantRegistrar registrar(TID_CALCACCEL, "OMP calcaccel");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcAccelOMPTask>(registrar, "calcaccel"); 
    }
    {
      TaskVariantRegistrar registrar(TID_ADVPOSFULL, "CPU advposfull");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::advPosFullTask>(registrar, "advposfull");
    }
    {
      TaskVariantRegistrar registrar(TID_ADVPOSFULL, "OMP advposfull");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::advPosFullOMPTask>(registrar, "advposfull");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCWORK, "CPU calcwork");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcWorkTask>(registrar, "calcwork");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCWORK, "OMP calcwork");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcWorkOMPTask>(registrar, "calcwork");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCWORKRATE, "CPU calcworkrate");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcWorkRateTask>(registrar, "calcworkrate");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCWORKRATE, "CPU calcworkrate");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcWorkRateOMPTask>(registrar, "calcworkrate");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCENERGY, "CPU calcenergy");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcEnergyTask>(registrar, "calcenergy");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCENERGY, "CPU calcenergy");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcEnergyOMPTask>(registrar, "calcenergy");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCDTNEW, "CPU calcdtnew");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<double, Hydro::calcDtNewTask>(registrar, "calcdtnew");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCDVOL, "CPU calcdvol");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<double, Hydro::calcDvolTask>(registrar, "calcdvol");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCDT, "CPU calcdt");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<double, Hydro::calcDtTask>(registrar, "calcdt");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCDTNEW, "OMP calcdtnew");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<double, Hydro::calcDtNewOMPTask>(registrar, "calcdtnew");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCDVOL, "OMP calcdvol");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<double, Hydro::calcDvolOMPTask>(registrar, "calcdvol");
    }
    {
      TaskVariantRegistrar registrar(TID_INITSUBRGN, "CPU init subrange");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::initSubrgnTask>(registrar, "init subrange");
    }
    {
      TaskVariantRegistrar registrar(TID_INITHYDRO, "CPU init hydro");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::initHydroTask>(registrar, "init hydro");
    }
    {
      TaskVariantRegistrar registrar(TID_INITRADIALVEL, "CPU init radial vel");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::initRadialVelTask>(registrar, "init radial vel");
    }
}
}; // namespace


Hydro::Hydro(
        const InputFile* inp,
        Mesh* m,
        Context ctxa,
        Runtime* runtimea)
        : mesh(m), ctx(ctxa), runtime(runtimea) {
    cfl = inp->getDouble("cfl", 0.6);
    cflv = inp->getDouble("cflv", 0.1);
    rinit = inp->getDouble("rinit", 1.);
    einit = inp->getDouble("einit", 0.);
    rinitsub = inp->getDouble("rinitsub", 1.);
    einitsub = inp->getDouble("einitsub", 0.);
    uinitradial = inp->getDouble("uinitradial", 0.);
    bcx = inp->getDoubleList("bcx", vector<double>());
    bcy = inp->getDoubleList("bcy", vector<double>());

    pgas = new PolyGas(inp, this);
    tts = new TTS(inp, this);
    qcs = new QCS(inp, this);

    const double2 vfixx = double2(1., 0.);
    const double2 vfixy = double2(0., 1.);
    if (mesh->parallel) {
      for (int i = 0; i < bcx.size(); ++i)
        bcs.push_back(new HydroBC(mesh, vfixx, bcx[i], true/*xplane*/));
      for (int i = 0; i < bcy.size(); ++i)
        bcs.push_back(new HydroBC(mesh, vfixy, bcy[i], false/*xplane*/));

      initParallel();
    } else {
      for (int i = 0; i < bcx.size(); ++i)
          bcs.push_back(new HydroBC(mesh, vfixx, mesh->getXPlane(bcx[i])));
      for (int i = 0; i < bcy.size(); ++i)
          bcs.push_back(new HydroBC(mesh, vfixy, mesh->getYPlane(bcy[i])));

      init();
    }
}


Hydro::~Hydro() {

    delete tts;
    delete qcs;
    for (int i = 0; i < bcs.size(); ++i) {
        delete bcs[i];
    }
}


void Hydro::init() {

    const int numpch = mesh->numpch;
    const int numzch = mesh->numzch;
    const int nump = mesh->nump;
    const int numz = mesh->numz;

    const double2* zx = mesh->zx;
    const double* zvol = mesh->zvol;

    // allocate arrays
    pu = alloc<double2>(nump);
    zm = alloc<double>(numz);
    zr = alloc<double>(numz);
    ze = alloc<double>(numz);
    zetot = alloc<double>(numz);
    zwrate = alloc<double>(numz);
    zp = alloc<double>(numz);

    // initialize hydro vars
    for (int zch = 0; zch < numzch; ++zch) {
        int zfirst = mesh->zchzfirst[zch];
        int zlast = mesh->zchzlast[zch];

        fill(&zr[zfirst], &zr[zlast], rinit);
        fill(&ze[zfirst], &ze[zlast], einit);
        fill(&zwrate[zfirst], &zwrate[zlast], 0.);

        const vector<double>& subrgn = mesh->subregion;
        if (!subrgn.empty()) {
            const double eps = 1.e-12;
            #pragma ivdep
            for (int z = zfirst; z < zlast; ++z) {
                if (zx[z].x > (subrgn[0] - eps) &&
                    zx[z].x < (subrgn[1] + eps) &&
                    zx[z].y > (subrgn[2] - eps) &&
                    zx[z].y < (subrgn[3] + eps)) {
                    zr[z] = rinitsub;
                    ze[z] = einitsub;
                }
            }
        }

        #pragma ivdep
        for (int z = zfirst; z < zlast; ++z) {
            zm[z] = zr[z] * zvol[z];
            zetot[z] = ze[z] * zm[z];
        }
    }  // for sch

    for (int pch = 0; pch < numpch; ++pch) {
        int pfirst = mesh->pchpfirst[pch];
        int plast = mesh->pchplast[pch];
        if (uinitradial != 0.)
            initRadialVel(uinitradial, pfirst, plast);
        else
            fill(&pu[pfirst], &pu[plast], double2(0., 0.));
    }  // for pch

    LogicalRegion& lrp = mesh->lrp;
    LogicalRegion& lrz = mesh->lrz;
    mesh->setField(lrp, FID_PU, pu, nump);
    mesh->setField(lrz, FID_ZM, zm, numz);
    mesh->setField(lrz, FID_ZR, zr, numz);
    mesh->setField(lrz, FID_ZE, ze, numz);
    mesh->setField(lrz, FID_ZETOT, zetot, numz);
    mesh->setField(lrz, FID_ZWRATE, zwrate, numz);

}


void Hydro::initParallel() {
  const LogicalRegion& lrp = mesh->lrp;
  const LogicalPartition lppprv = mesh->lppprv;
  const LogicalPartition lppmstr = mesh->lppmstr;
  const LogicalRegion& lrz = mesh->lrz;
  const LogicalPartition lpz = mesh->lpz;
  const IndexSpace is_piece = mesh->ispc;
  const vector<double>& subrgn = mesh->subregion;
  // fill zr
  {
    FillLauncher launcher(lrz, lrz, TaskArgument(&rinit, sizeof(rinit)));
    launcher.add_field(FID_ZR);
    runtime->fill_fields(ctx, launcher); 
  }
    
  // fill ze
  {
    FillLauncher launcher(lrz, lrz, TaskArgument(&einit, sizeof(einit)));
    launcher.add_field(FID_ZE);
    runtime->fill_fields(ctx, launcher);
  }

  // fill zwrate
  {
    const double zero = 0.;
    FillLauncher launcher(lrz, lrz, TaskArgument(&zero, sizeof(zero)));
    launcher.add_field(FID_ZWRATE);
    runtime->fill_fields(ctx, launcher);
  }

  // see if we have to initialize subrange
  if (!subrgn.empty())
  {
    const double eps = 1.e-12;
    InitSubrgnArgs args(subrgn, eps, rinitsub, einitsub); 
    IndexTaskLauncher launcher(TID_INITSUBRGN, is_piece,
        TaskArgument(&args, sizeof(args)), ArgumentMap());
    launcher.add_region_requirement(
        RegionRequirement(lpz, 0/*identity*/, READ_ONLY, EXCLUSIVE, lrz));
    launcher.add_field(0/*index*/, FID_ZX);
    launcher.add_region_requirement(
        RegionRequirement(lpz, 0/*identity*/, READ_WRITE, EXCLUSIVE, lrz));
    launcher.add_field(1/*index*/, FID_ZR);
    launcher.add_field(1/*index*/, FID_ZE);
    runtime->execute_index_space(ctx, launcher);
  }

  // compute zm and ze
  {
    IndexTaskLauncher launcher(TID_INITHYDRO, is_piece, 
                          TaskArgument(), ArgumentMap());
    launcher.add_region_requirement(
        RegionRequirement(lpz, 0/*identity*/, READ_ONLY, EXCLUSIVE, lrz));
    launcher.add_field(0/*index*/, FID_ZR);
    launcher.add_field(0/*index*/, FID_ZVOL);
    launcher.add_field(0/*index*/, FID_ZE);
    launcher.add_region_requirement(
        RegionRequirement(lpz, 0/*identity*/, WRITE_DISCARD, EXCLUSIVE, lrz));
    launcher.add_field(1/*index*/, FID_ZM);
    launcher.add_field(1/*index*/, FID_ZETOT);
    runtime->execute_index_space(ctx, launcher);
  }

  // Initialize the radial velocity
  if (uinitradial != 0.)
  {
    const double eps = 1.e-12;
    const InitRadialVelArgs args(uinitradial, eps);
    {
      IndexTaskLauncher launcher(TID_INITRADIALVEL, is_piece,
            TaskArgument(&args, sizeof(args)), ArgumentMap());
      launcher.add_region_requirement(
          RegionRequirement(lppprv, 0/*identity*/, READ_ONLY, EXCLUSIVE, lrp));
      launcher.add_field(0/*index*/, FID_PX);
      launcher.add_region_requirement(
          RegionRequirement(lppprv, 0/*identity*/, WRITE_DISCARD, EXCLUSIVE, lrp));
      launcher.add_field(1/*index*/, FID_PU);
      runtime->execute_index_space(ctx, launcher);
    }
    {
      IndexTaskLauncher launcher(TID_INITRADIALVEL, is_piece,
            TaskArgument(&args, sizeof(args)), ArgumentMap());
      launcher.add_region_requirement(
          RegionRequirement(lppmstr, 0/*identity*/, READ_ONLY, EXCLUSIVE, lrp));
      launcher.add_field(0/*index*/, FID_PX);
      launcher.add_region_requirement(
          RegionRequirement(lppmstr, 0/*identity*/, WRITE_DISCARD, EXCLUSIVE, lrp));
      launcher.add_field(1/*index*/, FID_PU);
      runtime->execute_index_space(ctx, launcher);
    }
  }
  else
  {
    const double2 zero2(0., 0.);
    FillLauncher launcher(lrp, lrp, TaskArgument(&zero2,sizeof(zero2)));
    launcher.add_field(FID_PU);
    runtime->fill_fields(ctx, launcher);
  }
}


void Hydro::initRadialVel(
        const double vel,
        const int pfirst,
        const int plast) {
    const double2* px = mesh->px;
    const double eps = 1.e-12;

    #pragma ivdep
    for (int p = pfirst; p < plast; ++p) {
        double pmag = length(px[p]);
        if (pmag > eps)
            pu[p] = vel * px[p] / pmag;
        else
            pu[p] = double2(0., 0.);
    }
}


Future Hydro::doCycle(
            Future f_dt,
            const int cycle,
            Predicate p_not_done) {

    LogicalRegion& lrp = mesh->lrp;
    LogicalRegion& lrs = mesh->lrs;
    LogicalRegion& lrz = mesh->lrz;
    LogicalPartition& lppprv = mesh->lppprv;
    LogicalPartition& lppmstr = mesh->lppmstr;
    LogicalPartition& lppshr = mesh->lppshr;
    LogicalPartition& lps = mesh->lps;
    LogicalPartition& lpz = mesh->lpz;
    //LogicalRegion& lrglb = mesh->lrglb;
    const IndexSpace& ispc= mesh->ispc;

    TaskArgument ta;
    ArgumentMap am;

    IndexCopyLauncher launchcfd(ispc);
    launchcfd.add_copy_requirements(
        RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz),
        RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcfd.add_src_field(0/*index*/, FID_ZVOL);
    launchcfd.add_dst_field(0/*index*/, FID_ZVOL0);
    launchcfd.predicate = p_not_done;
    runtime->issue_copy_operation(ctx, launchcfd);
    
    // begin hydro cycle
    double ffdargs[] = { 0. };
    IndexFillLauncher launchffd;
    launchffd.launch_space = ispc;
    launchffd.projection = 0;
    launchffd.argument = TaskArgument(ffdargs, sizeof(ffdargs));
    launchffd.predicate = p_not_done;

    double2 ffd2args[] = { double2(0., 0.) };
    IndexFillLauncher launchffd2;
    launchffd2.launch_space = ispc;
    launchffd2.projection = 0;
    launchffd2.argument = TaskArgument(ffd2args, sizeof(ffd2args));
    launchffd2.predicate = p_not_done;
    
    IndexTaskLauncher launchaph(TID_ADVPOSHALF, ispc, ta, am, p_not_done);
    launchaph.add_future(f_dt);
    // do point routines twice, once each for private and master
    // partitions
    for (int part = 0; part < 2; ++part) {
        LogicalPartition& lppcurr = (part == 0 ? lppprv : lppmstr);
        launchcfd.src_requirements[0] = 
                RegionRequirement(lppcurr, 0, READ_ONLY, EXCLUSIVE, lrp);
        launchcfd.add_src_field(0, FID_PX);
        launchcfd.dst_requirements[0] = 
                RegionRequirement(lppcurr, 0, WRITE_DISCARD, EXCLUSIVE, lrp);
        launchcfd.add_dst_field(0, FID_PX0);
        runtime->issue_copy_operation(ctx, launchcfd);

        // reuse copy launcher for different field
        launchcfd.src_requirements[0].privilege_fields.clear();
        launchcfd.src_requirements[0].instance_fields.clear();
        launchcfd.add_src_field(0, FID_PU);
        launchcfd.dst_requirements[0].privilege_fields.clear();
        launchcfd.dst_requirements[0].instance_fields.clear();
        launchcfd.add_dst_field(0, FID_PU0);
        runtime->issue_copy_operation(ctx, launchcfd);

        launchffd.partition = lppcurr;
        launchffd.parent = lrp;
        launchffd.fields.clear();
        launchffd.add_field(FID_PMASWT);
        runtime->fill_fields(ctx, launchffd);

        launchffd2.partition = lppcurr;
        launchffd2.parent = lrp;
        launchffd2.fields.clear();
        launchffd2.add_field(FID_PF);
        runtime->fill_fields(ctx, launchffd2);

        launchaph.region_requirements.clear();
        launchaph.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        READ_ONLY, EXCLUSIVE, lrp));
        launchaph.add_field(0, FID_PX0);
        launchaph.add_field(0, FID_PU0);
        launchaph.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        WRITE_DISCARD, EXCLUSIVE, lrp));
        launchaph.add_field(1, FID_PXP);
        // Only really need OpenMP for the private part
        if (part == 0)
          launchaph.tag |= PennantMapper::CRITICAL |
            PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
        else
          launchaph.tag &= ~(PennantMapper::PREFER_OMP);
        runtime->execute_index_space(ctx, launchaph);
    }  // for part

    IndexTaskLauncher launchcc(TID_CALCCTRS, ispc, ta, am, p_not_done);
    launchcc.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcc.add_field(0, FID_MAPSP1);
    launchcc.add_field(0, FID_MAPSP2);
    launchcc.add_field(0, FID_MAPSZ);
    launchcc.add_field(0, FID_MAPSP1REG);
    launchcc.add_field(0, FID_MAPSP2REG);
    launchcc.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcc.add_field(1, FID_ZNUMP);
    launchcc.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcc.add_field(2, FID_PXP);
    launchcc.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcc.add_field(3, FID_PXP);
    launchcc.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchcc.add_field(4, FID_EXP);
    launchcc.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcc.add_field(5, FID_ZXP);
    launchcc.tag |= PennantMapper::CRITICAL | 
      PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchcc);

    IndexTaskLauncher launchcv(TID_CALCVOLS, ispc, ta, am, p_not_done);
    launchcv.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcv.add_field(0, FID_MAPSP1);
    launchcv.add_field(0, FID_MAPSP2);
    launchcv.add_field(0, FID_MAPSZ);
    launchcv.add_field(0, FID_MAPSP1REG);
    launchcv.add_field(0, FID_MAPSP2REG);
    launchcv.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcv.add_field(1, FID_PXP);
    launchcv.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcv.add_field(2, FID_PXP);
    launchcv.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcv.add_field(3, FID_ZXP);
    launchcv.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchcv.add_field(4, FID_SAREAP);
    launchcv.add_field(4, FID_SVOLP);
    launchcv.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcv.add_field(5, FID_ZAREAP);
    launchcv.add_field(5, FID_ZVOLP);
    launchcv.tag |= PennantMapper::CRITICAL | 
      PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    Future f_cv = runtime->execute_index_space(ctx, launchcv, OPID_SUMINT);

    IndexTaskLauncher launchcsv(TID_CALCSURFVECS, ispc, ta, am, p_not_done);
    launchcsv.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcsv.add_field(0, FID_MAPSZ);
    launchcsv.add_field(0, FID_EXP);
    launchcsv.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcsv.add_field(1, FID_ZXP);
    launchcsv.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchcsv.add_field(2, FID_SSURFP);
    launchcsv.tag |= PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchcsv);

    IndexTaskLauncher launchcel(TID_CALCEDGELEN, ispc, ta, am, p_not_done);
    launchcel.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcel.add_field(0, FID_MAPSP1);
    launchcel.add_field(0, FID_MAPSP2);
    launchcel.add_field(0, FID_MAPSP1REG);
    launchcel.add_field(0, FID_MAPSP2REG);
    launchcel.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcel.add_field(1, FID_PXP);
    launchcel.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcel.add_field(2, FID_PXP);
    launchcel.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchcel.add_field(3, FID_ELEN);
    launchcel.tag |= PennantMapper::CRITICAL | 
      PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchcel);

    IndexTaskLauncher launchccl(TID_CALCCHARLEN, ispc, ta, am, p_not_done);
    launchccl.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchccl.add_field(0, FID_MAPSZ);
    launchccl.add_field(0, FID_SAREAP);
    launchccl.add_field(0, FID_ELEN);
    launchccl.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchccl.add_field(1, FID_ZNUMP);
    launchccl.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchccl.add_field(2, FID_ZDL);
    launchccl.tag |= PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchccl);

    IndexTaskLauncher launchcr(TID_CALCRHO, ispc, ta, am, p_not_done);
    launchcr.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcr.add_field(0, FID_ZM);
    launchcr.add_field(0, FID_ZVOLP);
    launchcr.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcr.add_field(1, FID_ZRP);
    launchcr.tag |= PennantMapper::CRITICAL | 
      PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchcr);

    IndexTaskLauncher launchccm(TID_CALCCRNRMASS, ispc, ta, am, p_not_done);
    launchccm.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchccm.add_field(0, FID_MAPSP1);
    launchccm.add_field(0, FID_MAPSP1REG);
    launchccm.add_field(0, FID_MAPSS3);
    launchccm.add_field(0, FID_MAPSZ);
    launchccm.add_field(0, FID_SMF);
    launchccm.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchccm.add_field(1, FID_ZRP);
    launchccm.add_field(1, FID_ZAREAP);
    launchccm.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_WRITE, EXCLUSIVE, lrp));
    launchccm.add_field(2, FID_PMASWT);
    launchccm.add_region_requirement(
            RegionRequirement(lppshr, 0, OPID_SUMDBL,
                    SIMULTANEOUS, lrp));
    launchccm.add_field(3, FID_PMASWT);
    launchccm.tag |= PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchccm);

    double cshargs[] = { pgas->gamma, pgas->ssmin };
    IndexTaskLauncher launchcsh(TID_CALCSTATEHALF, ispc,
            TaskArgument(cshargs, sizeof(cshargs)), am, p_not_done);
    launchcsh.add_future(f_dt);
    launchcsh.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcsh.add_field(0, FID_ZR);
    launchcsh.add_field(0, FID_ZVOLP);
    launchcsh.add_field(0, FID_ZVOL0);
    launchcsh.add_field(0, FID_ZE);
    launchcsh.add_field(0, FID_ZWRATE);
    launchcsh.add_field(0, FID_ZM);
    launchcsh.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcsh.add_field(1, FID_ZP);
    launchcsh.add_field(1, FID_ZSS);
    launchcsh.tag |= PennantMapper::CRITICAL | 
      PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchcsh);

    IndexTaskLauncher launchcfp(TID_CALCFORCEPGAS, ispc, ta, am, p_not_done);
    launchcfp.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcfp.add_field(0, FID_MAPSZ);
    launchcfp.add_field(0, FID_SSURFP);
    launchcfp.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcfp.add_field(1, FID_ZP);
    launchcfp.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchcfp.add_field(2, FID_SFP);
    launchcfp.tag |= PennantMapper::CRITICAL | 
      PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchcfp);

    double cftargs[] = { tts->alfa, tts->ssmin };
    IndexTaskLauncher launchcft(TID_CALCFORCETTS, ispc,
            TaskArgument(cftargs, sizeof(cftargs)), am, p_not_done);
    launchcft.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcft.add_field(0, FID_MAPSZ);
    launchcft.add_field(0, FID_SAREAP);
    launchcft.add_field(0, FID_SMF);
    launchcft.add_field(0, FID_SSURFP);
    launchcft.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcft.add_field(1, FID_ZAREAP);
    launchcft.add_field(1, FID_ZRP);
    launchcft.add_field(1, FID_ZSS);
    launchcft.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchcft.add_field(2, FID_SFT);
    launchcft.tag |= PennantMapper::CRITICAL | 
      PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchcft);

    IndexTaskLauncher launchscd(TID_SETCORNERDIV, ispc, ta, am, p_not_done);
    launchscd.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchscd.add_field(0, FID_MAPSZ);
    launchscd.add_field(0, FID_MAPSP1);
    launchscd.add_field(0, FID_MAPSP2);
    launchscd.add_field(0, FID_MAPSS3);
    launchscd.add_field(0, FID_MAPSP1REG);
    launchscd.add_field(0, FID_MAPSP2REG);
    launchscd.add_field(0, FID_EXP);
    launchscd.add_field(0, FID_ELEN);
    launchscd.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchscd.add_field(1, FID_ZNUMP);
    launchscd.add_field(1, FID_ZXP);
    launchscd.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchscd.add_field(2, FID_PXP);
    launchscd.add_field(2, FID_PU0);
    launchscd.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchscd.add_field(3, FID_PXP);
    launchscd.add_field(3, FID_PU0);
    launchscd.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchscd.add_field(4, FID_ZUC);
    launchscd.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchscd.add_field(5, FID_CAREA);
    launchscd.add_field(5, FID_CCOS);
    launchscd.add_field(5, FID_CDIV);
    launchscd.add_field(5, FID_CEVOL);
    launchscd.add_field(5, FID_CDU);
    launchscd.tag |= PennantMapper::CRITICAL | 
      PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchscd);

    double sqcfargs[] = { qcs->qgamma, qcs->q1, qcs->q2 };
    IndexTaskLauncher launchsqcf(TID_SETQCNFORCE, ispc,
            TaskArgument(sqcfargs, sizeof(sqcfargs)), am, p_not_done);
    launchsqcf.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchsqcf.add_field(0, FID_MAPSZ);
    launchsqcf.add_field(0, FID_MAPSP1);
    launchsqcf.add_field(0, FID_MAPSP2);
    launchsqcf.add_field(0, FID_MAPSS3);
    launchsqcf.add_field(0, FID_MAPSP1REG);
    launchsqcf.add_field(0, FID_MAPSP2REG);
    launchsqcf.add_field(0, FID_ELEN);
    launchsqcf.add_field(0, FID_CDIV);
    launchsqcf.add_field(0, FID_CDU);
    launchsqcf.add_field(0, FID_CEVOL);
    launchsqcf.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchsqcf.add_field(1, FID_ZRP);
    launchsqcf.add_field(1, FID_ZSS);
    launchsqcf.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchsqcf.add_field(2, FID_PU0);
    launchsqcf.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchsqcf.add_field(3, FID_PU0);
    launchsqcf.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchsqcf.add_field(4, FID_CRMU);
    launchsqcf.add_field(4, FID_CQE1);
    launchsqcf.add_field(4, FID_CQE2);
    launchsqcf.tag |= PennantMapper::CRITICAL | 
      PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchsqcf);

    IndexTaskLauncher launchsfq(TID_SETFORCEQCS, ispc, ta, am, p_not_done);
    launchsfq.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchsfq.add_field(0, FID_MAPSS4);
    launchsfq.add_field(0, FID_CAREA);
    launchsfq.add_field(0, FID_CQE1);
    launchsfq.add_field(0, FID_CQE2);
    launchsfq.add_field(0, FID_ELEN);
    launchsfq.add_region_requirement(
            RegionRequirement(lps, 0, READ_WRITE, EXCLUSIVE, lrs));
    launchsfq.add_field(1, FID_CCOS);
    launchsfq.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchsfq.add_field(2, FID_CW);
    launchsfq.add_field(2, FID_SFQ);
    launchsfq.tag |= PennantMapper::CRITICAL | 
      PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchsfq);

    double svdargs[] = { qcs->q1, qcs->q2 };
    IndexTaskLauncher launchsvd(TID_SETVELDIFF, ispc,
            TaskArgument(svdargs, sizeof(svdargs)), am, p_not_done);
    launchsvd.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchsvd.add_field(0, FID_MAPSZ);
    launchsvd.add_field(0, FID_MAPSP1);
    launchsvd.add_field(0, FID_MAPSP2);
    launchsvd.add_field(0, FID_MAPSP1REG);
    launchsvd.add_field(0, FID_MAPSP2REG);
    launchsvd.add_field(0, FID_ELEN);
    launchsvd.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchsvd.add_field(1, FID_ZSS);
    launchsvd.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchsvd.add_field(2, FID_PXP);
    launchsvd.add_field(2, FID_PU0);
    launchsvd.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchsvd.add_field(3, FID_PXP);
    launchsvd.add_field(3, FID_PU0);
    launchsvd.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchsvd.add_field(4, FID_ZTMP);
    launchsvd.add_field(4, FID_ZDU);
    launchsvd.tag |= PennantMapper::CRITICAL | 
      PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchsvd);

    IndexTaskLauncher launchscf(TID_SUMCRNRFORCE, ispc, ta, am, p_not_done);
    launchscf.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchscf.add_field(0, FID_MAPSP1);
    launchscf.add_field(0, FID_MAPSP1REG);
    launchscf.add_field(0, FID_MAPSS3);
    launchscf.add_field(0, FID_SFP);
    launchscf.add_field(0, FID_SFQ);
    launchscf.add_field(0, FID_SFT);
    launchscf.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_WRITE, EXCLUSIVE, lrp));
    launchscf.add_field(1, FID_PF);
    launchscf.add_region_requirement(
            RegionRequirement(lppshr, 0, OPID_SUMDBL2,
                    SIMULTANEOUS, lrp));
    launchscf.add_field(2, FID_PF);
    launchscf.tag |= PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchscf);

    // 4a. apply boundary conditions
    IndexTaskLauncher launchafbc(TID_APPLYFIXEDBC, ispc, ta, am, p_not_done);
    for (int i = 0; i < bcs.size(); ++i) {
        double2 afbcargs[1] = { bcs[i]->vfix };
        launchafbc.global_arg =
                TaskArgument(afbcargs, sizeof(afbcargs));
        launchafbc.region_requirements.clear();
        launchafbc.add_region_requirement(
                RegionRequirement(bcs[i]->lpb, 0,
                        READ_ONLY, EXCLUSIVE, bcs[i]->lrb));
        launchafbc.add_field(0, FID_MAPBP);
        launchafbc.add_field(0, FID_MAPBPREG);
        launchafbc.add_region_requirement(
                RegionRequirement(lppprv, 0,
                        READ_WRITE, EXCLUSIVE, lrp));
        launchafbc.add_field(1, FID_PF);
        launchafbc.add_field(1, FID_PU0);
        launchafbc.add_region_requirement(
                RegionRequirement(lppmstr, 0,
                        READ_WRITE, EXCLUSIVE, lrp, PennantMapper::PREFER_ZCOPY));
        launchafbc.add_field(2, FID_PF);
        launchafbc.add_field(2, FID_PU0);
        launchafbc.tag |= PennantMapper::CRITICAL | 
          PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
        runtime->execute_index_space(ctx, launchafbc);
    }

    // check for negative volumes on predictor step
    mesh->checkBadSides(cycle, f_cv, p_not_done);

    IndexTaskLauncher launchca(TID_CALCACCEL, ispc, ta, am, p_not_done);
    IndexTaskLauncher launchapf(TID_ADVPOSFULL, ispc, ta, am, p_not_done);
    launchapf.add_future(f_dt);
    launchapf.tag |= PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    // do point routines twice, once each for private and master
    // partitions
    for (int part = 0; part < 2; ++part) {
        LogicalPartition& lppcurr = (part == 0 ? lppprv : lppmstr);

        // 5. compute accelerations
        launchca.region_requirements.clear();
        launchca.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        READ_ONLY, EXCLUSIVE, lrp,
                        (part == 0) ? 0 : PennantMapper::PREFER_ZCOPY));
        launchca.add_field(0, FID_PF);
        launchca.add_field(0, FID_PMASWT);
        launchca.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        WRITE_DISCARD, EXCLUSIVE, lrp));
        launchca.add_field(1, FID_PAP);
        // Only really need OpenMP for the private part
        // But the shared part is the one on the critical path
        if (part == 0)
          launchca.tag |= PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
        else
        {
          launchca.tag &= ~(PennantMapper::PREFER_OMP);
          launchca.tag |= PennantMapper::CRITICAL;
        }
        runtime->execute_index_space(ctx, launchca);

        // ===== Corrector step =====
        // 6. advance mesh to end of time step
        launchapf.region_requirements.clear();
        launchapf.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        READ_ONLY, EXCLUSIVE, lrp));
        launchapf.add_field(0, FID_PX0);
        launchapf.add_field(0, FID_PU0);
        launchapf.add_field(0, FID_PAP);
        launchapf.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        WRITE_DISCARD, EXCLUSIVE, lrp));
        launchapf.add_field(1, FID_PX);
        launchapf.add_field(1, FID_PU);
        // Only really need OpenMP for the private part
        if (part == 0)
          launchca.tag |= PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
        else
        {
          launchca.tag &= ~(PennantMapper::PREFER_OMP);
          launchca.tag |= PennantMapper::CRITICAL;
        }
        runtime->execute_index_space(ctx, launchapf);
    }  // for part

    // 6a. compute new mesh geometry
    // reuse launchers from earlier, with corrector-step fields
    for (int r = 2; r < 6; ++r) {
        launchcc.region_requirements[r].privilege_fields.clear();
        launchcc.region_requirements[r].instance_fields.clear();
    }
    launchcc.add_field(2, FID_PX);
    launchcc.add_field(3, FID_PX);
    launchcc.add_field(4, FID_EX);
    launchcc.add_field(5, FID_ZX);
    runtime->execute_index_space(ctx, launchcc);

    for (int r = 1; r < 6; ++r) {
        launchcv.region_requirements[r].privilege_fields.clear();
        launchcv.region_requirements[r].instance_fields.clear();
    }
    launchcv.add_field(1, FID_PX);
    launchcv.add_field(2, FID_PX);
    launchcv.add_field(3, FID_ZX);
    launchcv.add_field(4, FID_SAREA);
    launchcv.add_field(4, FID_SVOL);
    launchcv.add_field(5, FID_ZAREA);
    launchcv.add_field(5, FID_ZVOL);
    f_cv = runtime->execute_index_space(ctx, launchcv, OPID_SUMINT);

    // 7. compute work
    IndexTaskLauncher launchcw(TID_CALCWORK, ispc, ta, am, p_not_done);
    launchcw.add_future(f_dt);
    launchcw.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcw.add_field(0, FID_MAPSP1);
    launchcw.add_field(0, FID_MAPSP2);
    launchcw.add_field(0, FID_MAPSZ);
    launchcw.add_field(0, FID_MAPSP1REG);
    launchcw.add_field(0, FID_MAPSP2REG);
    launchcw.add_field(0, FID_SFP);
    launchcw.add_field(0, FID_SFQ);
    launchcw.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcw.add_field(1, FID_PU);
    launchcw.add_field(1, FID_PU0);
    launchcw.add_field(1, FID_PXP);
    launchcw.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcw.add_field(2, FID_PU);
    launchcw.add_field(2, FID_PU0);
    launchcw.add_field(2, FID_PXP);
    launchcw.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcw.add_field(3, FID_ZW);
    launchcw.add_region_requirement(
            RegionRequirement(lpz, 0, READ_WRITE, EXCLUSIVE, lrz));
    launchcw.add_field(4, FID_ZETOT);
    launchcw.tag |= PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchcw);

    IndexTaskLauncher launchcwr(TID_CALCWORKRATE, ispc, ta, am, p_not_done);
    launchcwr.add_future(f_dt);
    launchcwr.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcwr.add_field(0, FID_ZVOL0);
    launchcwr.add_field(0, FID_ZVOL);
    launchcwr.add_field(0, FID_ZW);
    launchcwr.add_field(0, FID_ZP);
    launchcwr.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcwr.add_field(1, FID_ZWRATE);
    launchcwr.tag |= PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchcwr);

    // 8. update state variables
    IndexTaskLauncher launchce(TID_CALCENERGY, ispc, ta, am, p_not_done);
    launchce.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchce.add_field(0, FID_ZETOT);
    launchce.add_field(0, FID_ZM);
    launchce.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchce.add_field(1, FID_ZE);
    launchce.tag |= PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    runtime->execute_index_space(ctx, launchce);

    // reuse launcher from earlier, with corrector-step fields
    for (int r = 0; r < 2; ++r) {
        launchcr.region_requirements[r].privilege_fields.clear();
        launchcr.region_requirements[r].instance_fields.clear();
    }
    launchcr.add_field(0, FID_ZM);
    launchcr.add_field(0, FID_ZVOL);
    launchcr.add_field(1, FID_ZR);
    runtime->execute_index_space(ctx, launchcr);

    // 9.  compute timestep for next cycle
    IndexTaskLauncher launchdtnew(TID_CALCDTNEW, ispc, 
        TaskArgument(&cfl, sizeof(cfl)), am, p_not_done);
    launchdtnew.add_region_requirement(
        RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchdtnew.add_field(0, FID_ZDL);
    launchdtnew.add_field(0, FID_ZDU);
    launchdtnew.add_field(0, FID_ZSS);
    launchdtnew.tag |= PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    Future f_dtnew = runtime->execute_index_space(ctx, launchdtnew, OPID_MINDBL);

    IndexTaskLauncher launchdvol(TID_CALCDVOL, ispc, ta, am, p_not_done);
    launchdvol.add_region_requirement(
        RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchdvol.add_field(0, FID_ZVOL);
    launchdvol.add_field(0, FID_ZVOL0);
    launchdvol.tag |= PennantMapper::CRITICAL | 
      PennantMapper::PREFER_OMP | PennantMapper::PREFER_GPU;
    Future f_dvol = runtime->execute_index_space(ctx, launchdvol, OPID_MAXDBL);

    // Single task launch to compute the future result
    TaskLauncher launchcdt(TID_CALCDT, TaskArgument(&cflv, sizeof(cflv)));
    launchcdt.add_future(f_dt);
    launchcdt.add_future(f_dtnew);
    launchcdt.add_future(f_dvol);
    Future f_cdt = runtime->execute_task(ctx, launchcdt);

    // check for negative volumes on corrector step
    mesh->checkBadSides(cycle, f_cv, p_not_done);

    return f_cdt;
}


void Hydro::getFinalState() {
    mesh->getField(mesh->lrp, FID_PX, mesh->px, mesh->nump);
    mesh->getField(mesh->lrz, FID_ZP, zp, mesh->numz);
    mesh->getField(mesh->lrz, FID_ZE, ze, mesh->numz);
    mesh->getField(mesh->lrz, FID_ZR, zr, mesh->numz);
}


void Hydro::advPosHalfTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double dt = task->futures[0].get_result<double>();
    const double dth = 0.5 * dt;
    const AccessorRO<double2> acc_px0(regions[0], FID_PX0);
    const AccessorRO<double2> acc_pu0(regions[0], FID_PU0);
    const AccessorWD<double2> acc_pxp(regions[1], FID_PXP);

    const IndexSpace& isp = task->regions[0].region.get_index_space();
        
    for (PointIterator itp(runtime, isp); itp(); itp++)
    {
        const double2 x0 = acc_px0[*itp];
        const double2 u0 = acc_pu0[*itp];
        const double2 xp = x0 + dth * u0;
        acc_pxp[*itp] = xp;
    }
}


void Hydro::advPosHalfOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double dt = task->futures[0].get_result<double>();
    const double dth = 0.5 * dt;
    const AccessorRO<double2> acc_px0(regions[0], FID_PX0);
    const AccessorRO<double2> acc_pu0(regions[0], FID_PU0);
    const AccessorWD<double2> acc_pxp(regions[1], FID_PXP);

    const IndexSpace& isp = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectp = runtime->get_index_space_domain(isp);
    #pragma omp parallel for 
    for (coord_t p = rectp.lo[0]; p <= rectp.hi[0]; p++)
    {
        const double2 x0 = acc_px0[p];
        const double2 u0 = acc_pu0[p];
        const double2 xp = x0 + dth * u0;
        acc_pxp[p] = xp;
    }
}


void Hydro::calcRhoTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    FieldID fid_zm = task->regions[0].instance_fields[0];
    FieldID fid_zvol = task->regions[0].instance_fields[1];
    const AccessorRO<double> acc_zm(regions[0], fid_zm);
    const AccessorRO<double> acc_zvol(regions[0], fid_zvol);
    FieldID fid_zr = task->regions[1].instance_fields[0];
    const AccessorWD<double> acc_zr(regions[1], fid_zr);

    const IndexSpace& isz = task->regions[0].region.get_index_space();
    for (PointIterator itz(runtime, isz); itz(); itz++)
    {
        const double m = acc_zm[*itz];
        const double v = acc_zvol[*itz];
        const double r = m / v;
        acc_zr[*itz] = r;
    }
}


void Hydro::calcRhoOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    FieldID fid_zm = task->regions[0].instance_fields[0];
    FieldID fid_zvol = task->regions[0].instance_fields[1];
    const AccessorRO<double> acc_zm(regions[0], fid_zm);
    const AccessorRO<double> acc_zvol(regions[0], fid_zvol);
    FieldID fid_zr = task->regions[1].instance_fields[0];
    const AccessorWD<double> acc_zr(regions[1], fid_zr);

    const IndexSpace& isz = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    #pragma omp parallel for
    for (coord_t z = rectz.lo[0]; z <= rectz.hi[0]; z++)
    {
        const double m = acc_zm[z];
        const double v = acc_zvol[z];
        const double r = m / v;
        acc_zr[z] = r;
    }
}


void Hydro::calcCrnrMassTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<Pointer> acc_mapss3(regions[0], FID_MAPSS3);
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double> acc_smf(regions[0], FID_SMF);
    const AccessorRO<double> acc_zr(regions[1], FID_ZRP);
    const AccessorRO<double> acc_zarea(regions[1], FID_ZAREAP);
    const AccessorRW<double> acc_pmas_prv(regions[2], FID_PMASWT);
    const AccessorRD<SumOp<double> > acc_pmas_shr(regions[3], FID_PMASWT, OPID_SUMDBL);

    const IndexSpace& iss = task->regions[0].region.get_index_space();

    for (PointIterator its(runtime, iss); its(); its++)
    {
        const Pointer s = *its;
        const Pointer s3 = acc_mapss3[s];
        const Pointer z  = acc_mapsz[s];
        const Pointer p = acc_mapsp1[s];
        const int preg = acc_mapsp1reg[s];
        const double r = acc_zr[z];
        const double area = acc_zarea[z];
        const double mf = acc_smf[s];
        const double mf3 = acc_smf[s3];
        const double mwt = r * area * 0.5 * (mf + mf3);
        if (preg == 0)
            SumOp<double>::apply<true/*exclusive*/>(acc_pmas_prv[p], mwt);
        else
            acc_pmas_shr[p] <<= mwt;
    }
}


void Hydro::calcCrnrMassOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<Pointer> acc_mapss3(regions[0], FID_MAPSS3);
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double> acc_smf(regions[0], FID_SMF);
    const AccessorRO<double> acc_zr(regions[1], FID_ZRP);
    const AccessorRO<double> acc_zarea(regions[1], FID_ZAREAP);
    const AccessorRW<double> acc_pmas_prv(regions[2], FID_PMASWT);
    const AccessorRD<SumOp<double>,false/*exclusive*/> 
      acc_pmas_shr(regions[3], FID_PMASWT, OPID_SUMDBL);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    #pragma omp parallel for
    for (coord_t s = rects.lo[0]; s <= rects.hi[0]; s++)
    {
        const Pointer s3 = acc_mapss3[s];
        const Pointer z  = acc_mapsz[s];
        const Pointer p = acc_mapsp1[s];
        const int preg = acc_mapsp1reg[s];
        const double r = acc_zr[z];
        const double area = acc_zarea[z];
        const double mf = acc_smf[s];
        const double mf3 = acc_smf[s3];
        const double mwt = r * area * 0.5 * (mf + mf3);
        if (preg == 0)
            SumOp<double>::apply<false/*exclusive*/>(acc_pmas_prv[p], mwt);
        else
            acc_pmas_shr[p] <<= mwt;
    }
}


void Hydro::sumCrnrForceTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<Pointer> acc_mapss3(regions[0], FID_MAPSS3);
    const AccessorRO<double2> acc_sfp(regions[0], FID_SFP);
    const AccessorRO<double2> acc_sfq(regions[0], FID_SFQ);
    const AccessorRO<double2> acc_sft(regions[0], FID_SFT);
    const AccessorRW<double2> acc_pf_prv(regions[1], FID_PF);
    const AccessorRD<SumOp<double2> > acc_pf_shr(regions[2], FID_PF, OPID_SUMDBL2);

    const IndexSpace& iss = task->regions[0].region.get_index_space();

    for (PointIterator its(runtime, iss); its(); its++)
    {
        const Pointer s = *its;
        const Pointer s3 = acc_mapss3[s];
        const Pointer p = acc_mapsp1[s];
        const int preg = acc_mapsp1reg[s];
        const double2 sfp = acc_sfp[s];
        const double2 sfq = acc_sfq[s];
        const double2 sft = acc_sft[s];
        const double2 sfp3 = acc_sfp[s3];
        const double2 sfq3 = acc_sfq[s3];
        const double2 sft3 = acc_sft[s3];
        const double2 cf = (sfp + sfq + sft) - (sfp3 + sfq3 + sft3);
        if (preg == 0)
            SumOp<double2>::apply<true/*exclusive*/>(acc_pf_prv[p], cf);
        else
            acc_pf_shr[p] <<= cf;
    }
}


void Hydro::sumCrnrForceOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<Pointer> acc_mapss3(regions[0], FID_MAPSS3);
    const AccessorRO<double2> acc_sfp(regions[0], FID_SFP);
    const AccessorRO<double2> acc_sfq(regions[0], FID_SFQ);
    const AccessorRO<double2> acc_sft(regions[0], FID_SFT);
    const AccessorRW<double2> acc_pf_prv(regions[1], FID_PF);
    const AccessorRD<SumOp<double2>,false/*exclusive*/> 
      acc_pf_shr(regions[2], FID_PF, OPID_SUMDBL2);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    #pragma omp parallel for  
    for (coord_t s = rects.lo[0]; s <= rects.hi[0]; s++)
    {
        const Pointer s3 = acc_mapss3[s];
        const Pointer p = acc_mapsp1[s];
        const int preg = acc_mapsp1reg[s];
        const double2 sfp = acc_sfp[s];
        const double2 sfq = acc_sfq[s];
        const double2 sft = acc_sft[s];
        const double2 sfp3 = acc_sfp[s3];
        const double2 sfq3 = acc_sfq[s3];
        const double2 sft3 = acc_sft[s3];
        const double2 cf = (sfp + sfq + sft) - (sfp3 + sfq3 + sft3);
        if (preg == 0)
            SumOp<double2>::apply<false/*exclusive*/>(acc_pf_prv[p], cf);
        else
            acc_pf_shr[p] <<= cf;
    }
}


void Hydro::calcAccelTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<double2> acc_pf(regions[0], FID_PF);
    const AccessorRO<double> acc_pmass(regions[0], FID_PMASWT);
    const AccessorWD<double2> acc_pa(regions[1], FID_PAP);

    const double fuzz = 1.e-99;
    const IndexSpace& isp = task->regions[0].region.get_index_space();

    for (PointIterator itp(runtime, isp); itp(); itp++)
    {
        const double2 f = acc_pf[*itp];
        const double m = acc_pmass[*itp];
        const double2 a = f / max(m, fuzz);
        acc_pa[*itp] = a;
    }
}


void Hydro::calcAccelOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<double2> acc_pf(regions[0], FID_PF);
    const AccessorRO<double> acc_pmass(regions[0], FID_PMASWT);
    const AccessorWD<double2> acc_pa(regions[1], FID_PAP);

    const double fuzz = 1.e-99;
    const IndexSpace& isp = task->regions[0].region.get_index_space();
    // This will assert if its not dense
    const Rect<1> rectp = runtime->get_index_space_domain(isp);
    #pragma omp parallel for
    for (coord_t p = rectp.lo[0]; p <= rectp.hi[0]; p++)
    {
        const double2 f = acc_pf[p];
        const double m = acc_pmass[p];
        const double2 a = f / max(m, fuzz);
        acc_pa[p] = a;
    }
}


void Hydro::advPosFullTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double dt = task->futures[0].get_result<double>();

    const AccessorRO<double2> acc_px0(regions[0], FID_PX0);
    const AccessorRO<double2> acc_pu0(regions[0], FID_PU0);
    const AccessorRO<double2> acc_pa(regions[0], FID_PAP);
    const AccessorWD<double2> acc_px(regions[1], FID_PX);
    const AccessorWD<double2> acc_pu(regions[1], FID_PU);

    const IndexSpace& isp = task->regions[0].region.get_index_space();

    for (PointIterator itp(runtime, isp); itp(); itp++)
    {
        const double2 x0 = acc_px0[*itp];
        const double2 u0 = acc_pu0[*itp];
        const double2 a = acc_pa[*itp];
        const double2 u = u0 + dt * a;
        acc_pu[*itp] = u;
        const double2 x = x0 + dt * 0.5 * (u0 + u);
        acc_px[*itp] = x;
    }
}


void Hydro::advPosFullOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double dt = task->futures[0].get_result<double>();

    const AccessorRO<double2> acc_px0(regions[0], FID_PX0);
    const AccessorRO<double2> acc_pu0(regions[0], FID_PU0);
    const AccessorRO<double2> acc_pa(regions[0], FID_PAP);
    const AccessorWD<double2> acc_px(regions[1], FID_PX);
    const AccessorWD<double2> acc_pu(regions[1], FID_PU);

    const IndexSpace& isp = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectp = runtime->get_index_space_domain(isp);
    #pragma omp parallel for
    for (coord_t p = rectp.lo[0]; p <= rectp.hi[0]; p++)
    {
        const double2 x0 = acc_px0[p];
        const double2 u0 = acc_pu0[p];
        const double2 a = acc_pa[p];
        const double2 u = u0 + dt * a;
        acc_pu[p] = u;
        const double2 x = x0 + dt * 0.5 * (u0 + u);
        acc_px[p] = x;
    }
}


void Hydro::calcWorkTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double dt = task->futures[0].get_result<double>();

    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    const AccessorRO<double2> acc_sf(regions[0], FID_SFP);
    const AccessorRO<double2> acc_sf2(regions[0], FID_SFQ);
    const AccessorRO<double2> acc_pu0[2] = {
        AccessorRO<double2>(regions[1], FID_PU0),
        AccessorRO<double2>(regions[2], FID_PU0)
    };
    const AccessorRO<double2> acc_pu[2] = {
        AccessorRO<double2>(regions[1], FID_PU),
        AccessorRO<double2>(regions[2], FID_PU)
    };
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[1], FID_PXP),
        AccessorRO<double2>(regions[2], FID_PXP)
    };
    const AccessorRW<double> acc_zw(regions[3], FID_ZW);
    const AccessorRW<double> acc_zetot(regions[4], FID_ZETOT);

    // Compute the work done by finding, for each element/node pair,
    //   dwork= force * vavg
    // where force is the force of the element on the node
    // and vavg is the average velocity of the node over the time period

    const IndexSpace& isz = task->regions[3].region.get_index_space();

    for (PointIterator itz(runtime, isz); itz(); itz++)
      acc_zw[*itz] = 0.;

    const double dth = 0.5 * dt;

    const IndexSpace& iss = task->regions[0].region.get_index_space();
 
    for (PointIterator its(runtime, iss); its(); its++)
    {
        const Pointer s = *its;
        const Pointer p1 = acc_mapsp1[s];
        const int p1reg = acc_mapsp1reg[s];
        const Pointer p2 = acc_mapsp2[s];
        const int p2reg = acc_mapsp2reg[s];
        const Pointer z = acc_mapsz[s];
        const double2 sf = acc_sf[s];
        const double2 sf2 = acc_sf2[s];
        const double2 sftot = sf + sf2;
        const double2 pu01 = acc_pu0[p1reg][p1];
        const double2 pu1 = acc_pu[p1reg][p1];
        const double sd1 = dot(sftot, (pu01 + pu1));
        const double2 pu02 = acc_pu0[p2reg][p2];
        const double2 pu2 = acc_pu[p2reg][p2];
        const double sd2 = dot(-sftot, (pu02 + pu2));
        const double2 px1 = acc_px[p1reg][p1];
        const double2 px2 = acc_px[p2reg][p2];
        const double dwork = -dth * (sd1 * px1.x + sd2 * px2.x);

        acc_zetot[z] += dwork;
        acc_zw[z] += dwork;
    }
}


void Hydro::calcWorkOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double dt = task->futures[0].get_result<double>();

    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    const AccessorRO<double2> acc_sf(regions[0], FID_SFP);
    const AccessorRO<double2> acc_sf2(regions[0], FID_SFQ);
    const AccessorRO<double2> acc_pu0[2] = {
        AccessorRO<double2>(regions[1], FID_PU0),
        AccessorRO<double2>(regions[2], FID_PU0)
    };
    const AccessorRO<double2> acc_pu[2] = {
        AccessorRO<double2>(regions[1], FID_PU),
        AccessorRO<double2>(regions[2], FID_PU)
    };
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[1], FID_PXP),
        AccessorRO<double2>(regions[2], FID_PXP)
    };
    const AccessorRW<double> acc_zw(regions[3], FID_ZW);
    const AccessorRW<double> acc_zetot(regions[4], FID_ZETOT);

    // Compute the work done by finding, for each element/node pair,
    //   dwork= force * vavg
    // where force is the force of the element on the node
    // and vavg is the average velocity of the node over the time period

    const IndexSpace& isz = task->regions[3].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    #pragma omp parallel for
    for (coord_t z = rectz.lo[0]; z <= rectz.hi[0]; z++)
      acc_zw[z] = 0.;

    const double dth = 0.5 * dt;

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss); 
    #pragma omp parallel for
    for (coord_t s = rects.lo[0]; s <= rects.hi[0]; s++)
    {
        const Pointer p1 = acc_mapsp1[s];
        const int p1reg = acc_mapsp1reg[s];
        const Pointer p2 = acc_mapsp2[s];
        const int p2reg = acc_mapsp2reg[s];
        const Pointer z = acc_mapsz[s];
        const double2 sf = acc_sf[s];
        const double2 sf2 = acc_sf2[s];
        const double2 sftot = sf + sf2;
        const double2 pu01 = acc_pu0[p1reg][p1];
        const double2 pu1 = acc_pu[p1reg][p1];
        const double sd1 = dot(sftot, (pu01 + pu1));
        const double2 pu02 = acc_pu0[p2reg][p2];
        const double2 pu2 = acc_pu[p2reg][p2];
        const double sd2 = dot(-sftot, (pu02 + pu2));
        const double2 px1 = acc_px[p1reg][p1];
        const double2 px2 = acc_px[p2reg][p2];
        const double dwork = -dth * (sd1 * px1.x + sd2 * px2.x);

        SumOp<double>::apply<false/*exclusive*/>(acc_zetot[z], dwork);
        SumOp<double>::apply<false/*exclusive*/>(acc_zw[z], dwork);
    }
}



void Hydro::calcWorkRateTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double dt = task->futures[0].get_result<double>();

    const AccessorRO<double> acc_zvol0(regions[0], FID_ZVOL0);
    const AccessorRO<double> acc_zvol(regions[0], FID_ZVOL);
    const AccessorRO<double> acc_zw(regions[0], FID_ZW);
    const AccessorRO<double> acc_zp(regions[0], FID_ZP);
    const AccessorWD<double> acc_zwrate(regions[1], FID_ZWRATE);

    double dtinv = 1. / dt;

    const IndexSpace& isz = task->regions[0].region.get_index_space();

    for (PointIterator itz(runtime, isz); itz(); itz++)
    {
        const double zvol = acc_zvol[*itz];
        const double zvol0 = acc_zvol0[*itz];
        const double dvol = zvol - zvol0;
        const double zw = acc_zw[*itz];
        const double zp = acc_zp[*itz];
        const double zwrate = (zw + zp * dvol) * dtinv;
        acc_zwrate[*itz] = zwrate;
    }
}


void Hydro::calcWorkRateOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double dt = task->futures[0].get_result<double>();

    const AccessorRO<double> acc_zvol0(regions[0], FID_ZVOL0);
    const AccessorRO<double> acc_zvol(regions[0], FID_ZVOL);
    const AccessorRO<double> acc_zw(regions[0], FID_ZW);
    const AccessorRO<double> acc_zp(regions[0], FID_ZP);
    const AccessorWD<double> acc_zwrate(regions[1], FID_ZWRATE);

    double dtinv = 1. / dt;

    const IndexSpace& isz = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    #pragma omp parallel for
    for (coord_t z = rectz.lo[0]; z <= rectz.hi[0]; z++)
    {
        const double zvol = acc_zvol[z];
        const double zvol0 = acc_zvol0[z];
        const double dvol = zvol - zvol0;
        const double zw = acc_zw[z];
        const double zp = acc_zp[z];
        const double zwrate = (zw + zp * dvol) * dtinv;
        acc_zwrate[z] = zwrate;
    }
}


void Hydro::calcEnergyTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<double> acc_zetot(regions[0], FID_ZETOT);
    const AccessorRO<double> acc_zm(regions[0], FID_ZM);
    const AccessorWD<double> acc_ze(regions[1], FID_ZE);

    const double fuzz = 1.e-99;
    const IndexSpace& isz = task->regions[0].region.get_index_space();

    for (PointIterator itz(runtime, isz); itz(); itz++)
    {
        const double zetot = acc_zetot[*itz];
        const double zm = acc_zm[*itz];
        const double ze = zetot / (zm + fuzz);
        acc_ze[*itz] = ze;
    }
}


void Hydro::calcEnergyOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<double> acc_zetot(regions[0], FID_ZETOT);
    const AccessorRO<double> acc_zm(regions[0], FID_ZM);
    const AccessorWD<double> acc_ze(regions[1], FID_ZE);

    const double fuzz = 1.e-99;
    const IndexSpace& isz = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    #pragma omp parallel for
    for (coord_t z = rectz.lo[0]; z <= rectz.hi[0]; z++)
    {
        const double zetot = acc_zetot[z];
        const double zm = acc_zm[z];
        const double ze = zetot / (zm + fuzz);
        acc_ze[z] = ze;
    }
}


double Hydro::calcDtNewTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double cfl    = args[0];

    const AccessorRO<double> acc_zdl(regions[0], FID_ZDL);
    const AccessorRO<double> acc_zdu(regions[0], FID_ZDU);
    const AccessorRO<double> acc_zss(regions[0], FID_ZSS);

    // compute dt using Courant condition
    const double fuzz = 1.e-99;
    double dtnew = 1.e99;
    const IndexSpace& isz = task->regions[0].region.get_index_space();
    
    for (PointIterator itz(runtime, isz); itz(); itz++)
    {
        const double zdu = acc_zdu[*itz];
        const double zss = acc_zss[*itz];
        const double cdu = max(zdu, max(zss, fuzz));
        const double zdl = acc_zdl[*itz];
        const double zdthyd = zdl * cfl / cdu;
        dtnew = (zdthyd < dtnew ? zdthyd : dtnew);
    }

    return dtnew;
}


double Hydro::calcDvolTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {

    const AccessorRO<double> acc_zvol(regions[0], FID_ZVOL);
    const AccessorRO<double> acc_zvol0(regions[0], FID_ZVOL0);

    // compute dt using volume condition
    double dvovmax = 1.e-99;
    const IndexSpace& isz = task->regions[0].region.get_index_space();
    
    for (PointIterator itz(runtime, isz); itz(); itz++)
    {
        const double zvol = acc_zvol[*itz];
        const double zvol0 = acc_zvol0[*itz];
        const double zdvov = abs((zvol - zvol0) / zvol0);
        dvovmax = (zdvov > dvovmax ? zdvov : dvovmax);
    }

    return dvovmax;
}


double Hydro::calcDtTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double cflv   = args[0];
    const double dtlast = task->futures[0].get_result<double>();
    const double dtnew  = task->futures[1].get_result<double>();
    const double dvovmax = task->futures[2].get_result<double>();

    double dtrec = 1.e99;
    if (dtnew < dtrec) {
        dtrec = dtnew;
    }
    double dtnew2 = dtlast * cflv / dvovmax;
    if (dtnew2 < dtrec) {
        dtrec = dtnew2;
    }
    return dtrec;
}


double Hydro::calcDtNewOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double cfl    = args[0];

    const AccessorRO<double> acc_zdl(regions[0], FID_ZDL);
    const AccessorRO<double> acc_zdu(regions[0], FID_ZDU);
    const AccessorRO<double> acc_zss(regions[0], FID_ZSS);

    // compute dt using Courant condition
    const double fuzz = 1.e-99;
    double dtnew = 1.e99;
    const IndexSpace& isz = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz); 
    #pragma omp parallel
    {
      double local = 1.e99;
      #pragma omp for
      for (coord_t z = rectz.lo[0]; z <= rectz.hi[0]; z++)
      {
          const double zdu = acc_zdu[z];
          const double zss = acc_zss[z];
          const double cdu = max(zdu, max(zss, fuzz));
          const double zdl = acc_zdl[z];
          const double zdthyd = zdl * cfl / cdu;
          MinOp<double>::fold<true/*exclusive*/>(local, zdthyd);
      }
      MinOp<double>::fold<false/*exclusive*/>(dtnew, local);
    }
    return dtnew;
}


double Hydro::calcDvolOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {

    const AccessorRO<double> acc_zvol(regions[0], FID_ZVOL);
    const AccessorRO<double> acc_zvol0(regions[0], FID_ZVOL0);

    // compute dt using volume condition
    double dvovmax = 1.e-99;
    const IndexSpace& isz = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz); 
    #pragma omp parallel
    {
      double local = 1.e-99;
      #pragma omp for
      for (coord_t z = rectz.lo[0]; z <= rectz.hi[0]; z++)
      {
          const double zvol = acc_zvol[z];
          const double zvol0 = acc_zvol0[z];
          const double zdvov = abs((zvol - zvol0) / zvol0);
          MaxOp<double>::fold<true/*exclusive*/>(local, zdvov);
      }
      MaxOp<double>::fold<false/*exclusive*/>(dvovmax, local);
    }
    return dvovmax;
}


void Hydro::initSubrgnTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
  const InitSubrgnArgs *args = reinterpret_cast<InitSubrgnArgs*>(task->args);

  const AccessorRO<double2> acc_zx(regions[0], FID_ZX);

  const AccessorWD<double> acc_zr(regions[1], FID_ZR);
  const AccessorWD<double> acc_ze(regions[1], FID_ZE);

  IndexSpace isz = task->regions[0].region.get_index_space();
  for (PointIterator itr(runtime, isz); itr(); itr++)
  {
    const double2 zx = acc_zx[*itr]; 
    if (zx.x > (args->subrgn[0] - args->eps) &&
        zx.x < (args->subrgn[1] + args->eps) &&
        zx.y > (args->subrgn[2] - args->eps) &&
        zx.y < (args->subrgn[3] + args->eps)) {
        acc_zr[*itr] = args->rinitsub;
        acc_ze[*itr] = args->einitsub;
    }
  }
}


void Hydro::initHydroTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
  const AccessorRO<double> acc_zr(regions[0], FID_ZR);
  const AccessorRO<double> acc_zvol(regions[0], FID_ZVOL);
  const AccessorRO<double> acc_ze(regions[0], FID_ZE);

  const AccessorWD<double> acc_zm(regions[1], FID_ZM);
  const AccessorWD<double> acc_zetot(regions[1], FID_ZETOT);

  IndexSpace isz = task->regions[0].region.get_index_space();
  for (PointIterator itr(runtime, isz); itr(); itr++)
  {
    const double zm = acc_zr[*itr] * acc_zvol[*itr];
    acc_zm[*itr] = zm;
    acc_zetot[*itr] = acc_ze[*itr] * zm;
  }
}


void Hydro::initRadialVelTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
  const InitRadialVelArgs *args = reinterpret_cast<const InitRadialVelArgs*>(task->args);

  const AccessorRO<double2> acc_px(regions[0], FID_PX);
  const AccessorWD<double2>  acc_pu(regions[1], FID_PU);

  IndexSpace isp = task->regions[0].region.get_index_space();
  for (PointIterator itr(runtime, isp); itr(); itr++)
  {
    const double2 px = acc_px[*itr];
    const double pmag = length(px);
    if (pmag > args->eps)
      acc_pu[*itr] = args->vel * px / pmag;
    else
      acc_pu[*itr] = double2(0., 0.);
  }
}


