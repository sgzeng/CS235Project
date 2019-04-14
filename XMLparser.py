#!/usr/bin/env python
# -*- coding: UTF-8 -*-

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import os

dataMap = {}
dir = "./stage1_dataset/test"

class Action():
    def __init__(self, api_name, call_name, call_pid, call_time, ret_value, apiArg_list, exInfo_list):
        self.api_name = api_name
        self.call_name = call_name
        self.call_pid = call_pid
        self.call_time = call_time
        self.ret_value = ret_value
        self.apiArg_list = apiArg_list
        self.exInfo_list = exInfo_list


if ( __name__ == "__main__"):
    for filename in os.listdir(dir):
        if filename.endswith(".xml"):
            actionlist = []
            tree = ET.ElementTree(file = os.path.join(dir, filename))
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
            key = filename.replace(".xml","")
            dataMap[key] = actionlist

    # could be slow
    #for filename, actionlist in dataMap:
    #    print("%s: %d" % (filename, len(actionlist)))
