#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Translation and localization module.

**Project Name:**      MakeHuman

**Product Home Page:** http://www.makehumancommunity.org/

**Code Home Page:**    http://code.google.com/p/makehuman/

**Authors:**           Joel Palmius, Marc Flerackers, Glynn Clements

**Copyright(c):**      MakeHuman Team 2001-2020

**Licensing:**         AGPL3

    This file is part of MakeHuman Community (www.makehumancommunity.org).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


Abstract
--------

Language file loading and translation.
"""

from collections import OrderedDict
import json
import os

from getpath import getDataPath, getSysDataPath
import log


class Language(object):
    def __init__(self):
        self.language = None
        self.languageStrings = None
        self.missingStrings = OrderedDict()
        self.rtl = False

    def setLanguage(self, lang):
        self.languageStrings = None
        path = os.path.join(getSysDataPath("languages/"), lang + ".json")
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            try:
                self.languageStrings = json.loads(f.read())
            except Exception:
                print("Error in language file %s" % lang)
                log.error("Error in language file %s", lang, exc_info=True)
                self.languageStrings = None
        self.language = lang
        self.rtl = False
        if self.languageStrings and "__options__" in self.languageStrings:
            self.rtl = self.languageStrings["__options__"].get("rtl", False)

    def getLanguageString(self, string, appendData=None, appendFormat=None):
        if not string:
            return string
        if string == "%%s":
            return string
        if isinstance(string, list):
            if len(string) == 2:
                appendData = string[1]
                string = string[0]
            else:
                if len(string) == 3:
                    appendFormat = string[2]
                    appendData = string[1]
                    string = string[0]

        if self.languageStrings is None:
            result = string
        else:
            result = self.languageStrings.get(string)

        if result is not None:
            if appendData is not None:
                if appendFormat is not None:
                    result = result + (appendFormat % appendData)
                else:
                    result = result + appendData
            return result
        self.missingStrings[string] = None
        return string

    def dumpMissingStrings(self):
        if not self.language:
            return
        path = os.path.join(getSysDataPath("languages/"), self.language + ".missing")
        pathdir = os.path.dirname(path)
        if not os.path.isdir(pathdir):
            os.makedirs(pathdir)
        with open(path, "w", encoding="utf-8") as f:
            for string in self.missingStrings.keys():
                if self.language == "master":
                    f.write(
                        '"%s": "%s",\n'
                        % (
                            string.replace("\n", "\\n").encode("utf8"),
                            string.replace("\n", "\\n").encode("utf8"),
                        )
                    )
                else:
                    f.write('"%s": "",\n' % string.replace("\n", "\\n").encode("utf8"))


def getLanguages():
    """
    The languages available on this MH installation, by listing all .json
    files in the languages folder in user and system data path.
    """
    langDirFiles = os.listdir(getSysDataPath("languages"))
    try:
        langDirFiles = langDirFiles + os.listdir(getDataPath("languages"))
    except Exception:
        print("No user data path")
    return ["english"] + [
        os.path.basename(filename).replace(".json", "")
        for filename in sorted(langDirFiles)
        if filename.split(os.extsep)[-1] == "json"
    ]


language = Language()
