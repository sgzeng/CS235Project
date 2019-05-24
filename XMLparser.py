#!/usr/bin/env python
# -*- coding: UTF-8 -*-

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import os

dataMap = {}
directory = "./stage1_dataset/test"

class Action():
    def __init__(self, api_name, call_name, call_pid, call_time, ret_value, apiArg_list, exInfo_list):
        self.api_name = api_name
        self.call_name = call_name
        self.call_pid = call_pid
        self.call_time = call_time
        self.ret_value = ret_value
        self.apiArg_list = apiArg_list
        self.exInfo_list = exInfo_list

def _findActionList(filePath):
    actionlist = []
    tree = ET.ElementTree(file = filePath)
    for elem in tree.iterfind('file_list/file/start_boot/action_list/action'):
        apiArg_list = []
        exInfo_list = []
        for argInfo in elem.iter():
            if argInfo.tag == "apiArg_list":
                for arg in argInfo:
                    apiArg_list.append(arg.attrib["value"])
            if argInfo.tag == "exInfo_list":
                for arg in argInfo:
                    exInfo_list.append(arg.attrib["value"])
        action = Action(elem.attrib["api_name"],
                        elem.attrib["call_name"],
                        elem.attrib["call_pid"],
                        elem.attrib["call_time"],
                        elem.attrib["ret_value"],
                        apiArg_list,
                        exInfo_list)
        actionlist.append(action)
    return actionlist

def parseFile(fileName, dir=directory):
    filePath = os.path.join(dir, fileName)
    if not os.path.exists(filePath):
        return None
    return _findActionList(filePath)


def parseDir(dir=directory):
    for fileName in os.listdir(dir):
        if fileName.endswith(".xml"):
            actionlist = _findActionList(os.path.join(dir, fileName))
            yield actionlist

# usage example
# if ( __name__ == "__main__"):
#     fileName = "ff8a1943d5b51c7182a521212273eb2ff72487637f1d314fb21042750ffe79cd.xml"
#     actionlist = parseFile(fileName)
#     print("%s: %d" % (fileName, len(actionlist)))

#     for actionlist in parseDir():
#         # do something
