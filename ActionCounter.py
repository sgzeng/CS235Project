#!/usr/bin/python
# -*- coding: UTF-8 -*-

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import os

dataMap = {}
dir = "./stage1_dataset/test"

class Action():
    def __init__(self, api_name, call_name, call_pid, call_time, ret_value):
        self.api_name = api_name
        self.call_name = call_name
        self.call_pid = call_pid
        self.call_time = call_time
        self.ret_value = ret_value


if ( __name__ == "__main__"):
    for filename in os.listdir(dir):
        if filename.endswith(".xml"):
            actionlist = []
            tree = ET.ElementTree(file= dir+filename)
            for elem in tree.iterfind('file_list/file/start_boot/action_list/action'):
                action = Action(elem.attrib["api_name"],
                    elem.attrib["call_name"],
                    elem.attrib["call_pid"],
                    elem.attrib["call_time"],
                    elem.attrib["ret_value"])
                actionlist.append(action)
            key = filename.replace(".xml","")
            dataMap[key] = actionlist
            print("%s: %d" % (filename, len(actionlist)))

    # could be slow
    #for filename, actionlist in dataMap:
    #    print("%s: %d" % (filename, len(actionlist)))
