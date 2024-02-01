/*
 * WriteXY.hh
 *
 *  Created on: Dec 16, 2013
 *      Author: cferenba
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef WRITEXY_HH_
#define WRITEXY_HH_

#include <string>

// forward declarations
class Mesh;


class WriteXY {
public:

    WriteXY();
    ~WriteXY();

    void write(
            const std::string& basename,
            const double* zr,
            const double* ze,
            const double* zp,
            const int numz);

};


#endif /* WRITEXY_HH_ */
