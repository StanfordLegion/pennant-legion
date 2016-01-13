/*
 * WriteXY.cc
 *
 *  Created on: Dec 16, 2013
 *      Author: cferenba
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "WriteXY.hh"

#include <fstream>
#include <iomanip>

#include "Mesh.hh"

using namespace std;


WriteXY::WriteXY(Mesh* m) : mesh(m) {}

WriteXY::~WriteXY() {}


void WriteXY::write(
        const string& basename,
        const double* zr,
        const double* ze,
        const double* zp) {

    const int numz = mesh->numz;

    string xyname = basename + ".xy";
    ofstream ofs(xyname.c_str());
    ofs << scientific << setprecision(8);
    ofs << "#  zr" << endl;
    for (int z = 0; z < numz; ++z) {
        ofs << setw(5) << (z + 1) << setw(18) << zr[z] << endl;
    }
    ofs << "#  ze" << endl;
    for (int z = 0; z < numz; ++z) {
        ofs << setw(5) << (z + 1) << setw(18) << ze[z] << endl;
    }
    ofs << "#  zp" << endl;
    for (int z = 0; z < numz; ++z) {
        ofs << setw(5) << (z + 1) << setw(18) << zp[z] << endl;
    }
    ofs.close();

}

