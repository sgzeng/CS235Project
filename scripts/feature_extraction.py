#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from XMLparser import *
import sys
import os
import pandas as pd

def computeFuture(actionlist):
    api_cnt = len(actionlist)
    apis = set()
    call_pids = set()
    ret_value_equals0_cnt = 0
    return_values = set()
    for action in actionlist:
        apis.add(action.api_name)
        call_pids.add(action.call_pid)
        return_values.add(action.ret_value)
        if action.ret_value == 0:
            ret_value_equals0_cnt+=1
    api_distinct_cnt = len(apis)
    call_pid_distinct_cnt = len(call_pids)
    ret_value_distinct_cnt = len(ret_value_distinct_cnt)
    return [api_cnt, api_distinct_cnt, call_pid_distinct_cnt, ret_value_equals0_cnt, ret_value_distinct_cnt]


def makeCSV(rows, attributes, outputPath):
    df = pd.DataFrame(rows)
    columns = []
    for colNum in attributes:
        columns.append(colNum)
    df.columns = columns
    print('result has been written to ' + outputPath)
    df.to_csv(outputPath)


if __name__ == '__main__':
    attributes = ['api_cnt', 'api_distinct_cnt', 'call_pid_distinct_cnt', 'ret_value_equals0_cnt', 'ret_value_distinct_cnt']
    rows = []
    for actionlist in parseDir():
        rows.append(computeFuture(actionlist))
    makeCSV(rows, attributes, './')
