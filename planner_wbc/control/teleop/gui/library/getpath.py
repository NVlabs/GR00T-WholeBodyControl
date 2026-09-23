#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
**Project Name:**      MakeHuman

**Product Home Page:** http://www.makehumancommunity.org/

**Github Code Home Page:**    https://github.com/makehumancommunity/

**Authors:**           Jonas Hauquier, Glynn Clements, Joel Palmius, Marc Flerackers

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

Utility module for finding the user home path.
"""

import os
import sys

from library.xdg_parser import XDG_PATHS

__home_path = None

# Search for an optional configuration file, providing another location for the home folder.
# The encoding of the file must be utf-8 and an absolute path is expected.

if sys.platform.startswith("linux"):

    configFile = os.path.expanduser("~/.config/makehuman.conf")

elif sys.platform.startswith("darwin"):

    configFile = os.path.expanduser("~/Library/Application Support/MakeHuman/makehuman.conf")

elif sys.platform.startswith("win32"):

    configFile = os.path.join(os.getenv("LOCALAPPDATA", ""), "makehuman.conf")

else:

    configFile = ""

configPath = ""

if os.path.isfile(configFile):
    with open(configFile, "r", encoding="utf-8") as f:
        configPath = f.readline().strip()

        if os.path.isdir(configPath):
            __home_path = os.path.normpath(configPath).replace("\\", "/")


def pathToUnicode(path):
    """
    Unicode representation of the filename.
    Bytes is decoded with the codeset used by the filesystem of the operating
    system.
    Unicode representations of paths are fit for use in GUI.
    """

    if isinstance(path, bytes):
        # Approach for bytes string type
        try:
            return str(path, "utf-8")
        except UnicodeDecodeError:
            pass
        try:
            return str(path, sys.getfilesystemencoding())
        except UnicodeDecodeError:
            pass
        try:
            return str(path, sys.getdefaultencoding())
        except UnicodeDecodeError:
            pass
        try:
            import locale

            return str(path, locale.getpreferredencoding())
        except UnicodeDecodeError:
            return path
    else:
        return path


def formatPath(path):
    if path is None:
        return None
    return pathToUnicode(os.path.normpath(path).replace("\\", "/"))


def canonicalPath(path):
    """
    Return canonical name for location specified by path.
    Useful for comparing paths.
    """
    return formatPath(os.path.realpath(path))


def localPath(path):
    """
    Returns the path relative to the MH program directory,
    i.e. the inverse of canonicalPath.
    """
    path = os.path.realpath(path)
    root = os.path.realpath(getSysPath())
    return formatPath(os.path.relpath(path, root))


def getHomePath():
    """
    Find the user home path.
    Note: If you are looking for MakeHuman data, you probably want getPath()!
    """
    # Cache the home path
    global __home_path

    # The environment variable MH_HOME_LOCATION will supersede any other settings for the home folder.
    alt_home_path = os.environ.get("MH_HOME_LOCATION", "")
    if os.path.isdir(alt_home_path):
        __home_path = formatPath(alt_home_path)

    if __home_path is not None:
        return __home_path

    # Windows
    if sys.platform.startswith("win"):
        import winreg

        keyname = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
        # name = 'Personal'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, keyname) as k:
            try:
                value, type_ = winreg.QueryValueEx(k, "Personal")
            except FileNotFoundError:
                value, type_ = r"%USERPROFILE%\Documents", winreg.REG_EXPAND_SZ
            if type_ == winreg.REG_EXPAND_SZ:
                __home_path = formatPath(winreg.ExpandEnvironmentStrings(value))
                return __home_path
            elif type_ == winreg.REG_SZ:
                __home_path = formatPath(value)
                return __home_path
            else:
                raise RuntimeError("Couldn't determine user folder")

    # Linux
    elif sys.platform.startswith("linux"):
        doc_folder = XDG_PATHS.get("DOCUMENTS", "")
        if os.path.isdir(doc_folder):
            __home_path = doc_folder
        else:
            __home_path = pathToUnicode(os.path.expanduser("~"))

    # Darwin
    else:
        __home_path = os.path.expanduser("~")

    return __home_path


def getPath(subPath=""):
    """
    Get MakeHuman folder that contains per-user files, located in the user home
    path.
    """
    path = getHomePath()

    # Windows
    if sys.platform.startswith("win"):
        path = os.path.join(path, "makehuman")

    # MAC OSX
    elif sys.platform.startswith("darwin"):
        path = os.path.join(path, "Documents")
        path = os.path.join(path, "MakeHuman")

    # Unix/Linux
    else:
        path = os.path.join(path, "makehuman")

    path = os.path.join(path, "v1py3")

    if subPath:
        path = os.path.join(path, subPath)

    return formatPath(path)


def getDataPath(subPath=""):
    """
    Path to per-user data folder, should always be the same as getPath('data').
    """
    if subPath:
        path = getPath(os.path.join("data", subPath))
    else:
        path = getPath("data")
    return formatPath(path)


def getSysDataPath(subPath=""):
    """
    Path to the data folder that is installed with MakeHuman system-wide.
    NOTE: do NOT assume that getSysPath("data") == getSysDataPath()!
    """
    if subPath:
        path = getSysPath(os.path.join("data", subPath))
    else:
        path = getSysPath("data")
    return formatPath(path)


def getSysPath(subPath=""):
    """
    Path to the system folder where MakeHuman is installed (it is possible that
    data is stored in another path).
    Writing to this folder or modifying this data usually requires admin rights,
    contains system-wide data (for all users).
    """
    if subPath:
        path = os.path.join(".", subPath)
    else:
        path = "."
    return formatPath(path)


def _allnamesequal(name):
    return all(n == name[0] for n in name[1:])


def commonprefix(paths, sep="/"):
    """
    Implementation of os.path.commonprefix that works as you would expect.

    Source: http://rosettacode.org/wiki/Find_Common_Directory_Path#Python
    """
    from itertools import takewhile

    bydirectorylevels = list(zip(*[p.split(sep) for p in paths]))
    return sep.join(x[0] for x in takewhile(_allnamesequal, bydirectorylevels))


def isSubPath(subpath, path):
    """
    Verifies whether subpath is within path.
    """
    subpath = canonicalPath(subpath)
    path = canonicalPath(path)
    return commonprefix([subpath, path]) == path


def isSamePath(path1, path2):
    """
    Determines whether two paths point to the same location.
    """
    return canonicalPath(path1) == canonicalPath(path2)


def getRelativePath(path, relativeTo=[getDataPath(), getSysDataPath()], strict=False):
    """
    Return a relative file path, relative to one of the specified search paths.
    First valid path is returned, so order in which search paths are given matters.
    """
    if not isinstance(relativeTo, list):
        relativeTo = [relativeTo]

    relto = None
    for p in relativeTo:
        if isSubPath(path, p):
            relto = p
    if relto is None:
        if strict:
            return None
        else:
            return path

    relto = os.path.abspath(os.path.realpath(relto))
    path = os.path.abspath(os.path.realpath(path))
    rpath = os.path.relpath(path, relto)
    return formatPath(rpath)


def findFile(relPath, searchPaths=[getDataPath(), getSysDataPath()], strict=False):
    """
    Inverse of getRelativePath: find an absolute path from specified relative
    path in one of the search paths.
    First occurence is returned, so order in which search paths are given matters.
    Note: does NOT treat the path as relative to the current working dir, unless
    you explicitly specify '.' as one of the searchpaths.
    """
    if not isinstance(searchPaths, list):
        searchPaths = [searchPaths]

    for dataPath in searchPaths:
        path = os.path.join(dataPath, relPath)
        if os.path.isfile(path):
            return formatPath(path)

    if strict:
        return None
    else:
        return relPath


def thoroughFindFile(filename, searchPaths=[], searchDefaultPaths=True):
    """
    Extensively search the data paths to find a file with matching filename in
    as much cases as possible. If file is found, returns absolute filename.
    If nothing is found return the most probable filename.
    """
    # Ensure unix style path
    filename.replace("\\", "/")

    if not isinstance(searchPaths, list):
        searchPaths = [searchPaths]

    if searchDefaultPaths:
        # Search in user / sys data, and user / sys root folders
        searchPaths = list(searchPaths)
        searchPaths.extend([getDataPath(), getSysDataPath(), getPath(), getSysPath()])

    path = findFile(filename, searchPaths, strict=True)
    if path:
        return canonicalPath(path)

    # Treat as absolute path or search relative to application path
    if os.path.isfile(filename):
        return canonicalPath(filename)

    # Strip leading data/ folder if present (for the scenario where sysDataPath is not in sysPath)
    if filename.startswith("data/"):
        result = thoroughFindFile(filename[5:], searchPaths, False)
        if os.path.isfile(result):
            return result

    # Nothing found
    return formatPath(filename)


def search(paths, extensions, recursive=True, mutexExtensions=False):
    """
    Search for files with specified extensions in specified paths.
    If mutexExtensions is True, no duplicate files with only differing extension
    will be returned. Instead, only the file with highest extension precedence
    (extensions occurs earlier in the extensions list) is kept.
    """
    if isinstance(paths, str):
        paths = [paths]
    if isinstance(extensions, str):
        extensions = [extensions]
    extensions = [e[1:].lower() if e.startswith(".") else e.lower() for e in extensions]

    if mutexExtensions:
        discovered = dict()

        def _aggregate_files_mutexExt(filepath):
            basep, ext = os.path.splitext(filepath)
            ext = ext[1:]
            if basep in discovered:
                if extensions.index(ext) < extensions.index(discovered[basep]):
                    discovered[basep] = ext
            else:
                discovered[basep] = ext

    if recursive:
        for path in paths:
            for root, dirs, files in os.walk(path):
                for f in files:
                    ext = os.path.splitext(f)[1][1:].lower()
                    if ext in extensions:
                        if mutexExtensions:
                            _aggregate_files_mutexExt(os.path.join(root, f))
                        else:
                            yield pathToUnicode(os.path.join(root, f))
    else:
        for path in paths:
            if not os.path.isdir(path):
                continue
            for f in os.listdir(path):
                f = os.path.join(path, f)
                if os.path.isfile(f):
                    ext = os.path.splitext(f)[1][1:].lower()
                    if ext in extensions:
                        if mutexExtensions:
                            _aggregate_files_mutexExt(f)
                        else:
                            yield pathToUnicode(f)

    if mutexExtensions:
        for f in ["%s.%s" % (p, e) for p, e in list(discovered.items())]:
            yield pathToUnicode(f)


def getJailedPath(filepath, relativeTo, jailLimits=[getDataPath(), getSysDataPath()]):
    """
    Get a path to filepath, relative to relativeTo path, confined within the
    jailLimits folders. Returns None if the path would fall outside of the jail.
    This is a portable path which can be used for distributing eg. materials
    (texture paths are portable).
    Returns None if the filepath falls outside of the jail folders. Returns
    a path to filename relative to relativeTo path if it is a subpath of it,
    else returns a path relative to the jailLimits.
    """

    def _withinJail(path):
        for j in jailLimits:
            if isSubPath(path, j):
                return True
        return False

    # These paths may become messed up when using symlinks for user home.
    # Make sure we use the same paths when calculating relative paths.
    filepath = os.path.realpath(filepath)

    if relativeTo is str:
        relativeTo = os.path.realpath(relativeTo)

    output = None

    if _withinJail(filepath):
        relPath = getRelativePath(filepath, relativeTo, strict=True)
        if relPath:
            output = relPath
        else:
            output = getRelativePath(filepath, jailLimits)
    return output
