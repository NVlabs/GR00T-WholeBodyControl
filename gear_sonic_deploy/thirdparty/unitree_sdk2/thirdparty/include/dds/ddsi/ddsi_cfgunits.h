/*
 * Copyright(c) 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSI_CFGUNITS_H
#define DDSI_CFGUNITS_H

static const struct cfgunit cfgunits[] = {
  UNIT("bandwidth",
    DESCRIPTION(
      "<p>The unit must be specified explicitly. Recognised units: "
      "<i>X</i>b/s, <i>X</i>bps for bits/s or <i>X</i>B/s, <i>X</i>Bps for "
      "bytes/s; where <i>X</i> is an optional prefix: k for 10<sup>3</sup>, "
      "Ki for 2<sup>10</sup>, M for 10<sup>6</sup>, Mi for 2<sup>20</sup>, "
      "G for 10<sup>9</sup>, Gi for 2<sup>30</sup>.</p>"),
    PATTERN(
      "0|(\\d+(\\.\\d*)?([Ee][\\-+]?\\d+)?|\\.\\d+([Ee][\\-+]?\\d+)?) *([kMG]i?)?[Bb][p/]s")),
  UNIT("duration",
    DESCRIPTION(
      "<p>The unit must be specified explicitly. Recognised units: ns, us, ms, "
      "s, min, hr, day.</p>"),
    PATTERN(
      "0|(\\d+(\\.\\d*)?([Ee][\\-+]?\\d+)?|\\.\\d+([Ee][\\-+]?\\d+)?) *([num]?s|min|hr|day)")),
  UNIT("duration_inf",
    DESCRIPTION(
      "<p>Valid values are finite durations with an explicit unit or the "
      "keyword 'inf' for infinity. Recognised units: ns, us, ms, s, min, hr, "
      "day.</p>"),
    PATTERN(
      "inf|0|(\\d+(\\.\\d*)?([Ee][\\-+]?\\d+)?|\\.\\d+([Ee][\\-+]?\\d+)?) *([num]?s|min|hr|day)")),
  UNIT("memsize",
    DESCRIPTION(
      "<p>The unit must be specified explicitly. Recognised units: B (bytes), "
      "kB & KiB (2<sup>10</sup> bytes), MB & MiB (2<sup>20</sup> bytes), GB & "
      "GiB (2<sup>30</sup> bytes).</p>"),
    PATTERN(
      "0|(\\d+(\\.\\d*)?([Ee][\\-+]?\\d+)?|\\.\\d+([Ee][\\-+]?\\d+)?) *([kMG]i?)?B")),
  END_MARKER
};

#endif /* DDSI_CFGUNITS_H */
