#!/usr/bin/env python3
import os, io, argparse, datetime
import pymysql
import pandas as pd
from sshtunnel import SSHTunnelForwarder

def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents

def parse_args():
    parser = argparse.ArgumentParser(description='Parse results from tf benchmark runs')
    parser.add_argument('filename', type=str, help='Log file to prase or directory containing log files')
    args = parser.parse_args()
    files = []
    if os.path.isdir(args.filename):
        all_files = os.listdir(args.filename)
        for name in all_files:
            if not 'log' in name:
                continue
            files.append(os.path.join(args.filename, name))
    else:
        files = [args.filename]
    args.files = files
    return args

def main():
    args = parse_args()
    #results = []
    tests = []
    kernels=[]
    tflops=[]
    #parse results, get the Tflops value for "Best Perf" kernels
    glue=""
    for filename in args.files:
        for line in open(filename):
            if 'Branch name' in line:
                lst=line.split()
                #print("lst=",lst)
                branch_name=lst[3]

    for filename in args.files:
        for line in open(filename):
            if 'Best Perf' in line:
                lst=line.split()
                #print("lst=",lst)
                #results.append(print_to_string(glue.join(lst[8:]),lst[4]))
                tests.append(glue.join(lst[4:25]))
                kernels.append(glue.join(lst[32:]))
                tflops.append(lst[28])

    #print("results:",results)
    #print("kernels:",kernels)
    #print("tflops:",tflops)
    #sort results
    print("Number of tests:",len(tests))

    print("Branch name:",branch_name)

    sorted_tests = sorted(tests)
    print("sorted tests:",sorted_tests)
    sorted_tflops = [x for _,x in sorted(zip(tests,tflops))]
    print("sorted tflops:",sorted_tflops)
    sorted_kernels = [x for _,x in sorted(zip(tests,kernels))]
    print("sorted kernels:",sorted_kernels)

    user_name=os.environ["user_name"]
    #print("user_name=",user_name)
    password=os.environ["password"]
    #print("password=",password)
    hostname=os.environ["hostname"]
    print("hostname=",hostname)
    db_name=os.environ["db_name"]
    print("db_name=",db_name)
    print("now=",datetime.datetime.now())

    sql_hostname = '127.0.0.1'
    sql_username = user_name
    sql_password = password
    sql_main_database = 'miopen_perf'
    sql_port = 3306
    ssh_host = hostname
    ssh_user = user_name
    ssh_port = 20057

    with SSHTunnelForwarder(
            (ssh_host, ssh_port),
            ssh_username=ssh_user,
            ssh_password=password,
            remote_bind_address=(sql_hostname, sql_port)) as tunnel:
        conn = pymysql.connect(host='127.0.0.1', user=sql_username,
            passwd=sql_password, db=sql_main_database,
            port=tunnel.local_bind_port)
        query = '''SELECT VERSION();'''
        data = pd.read_sql_query(query, conn)
        print("data=",data)

        #read baseline results for the latest develop branch
        query = '''SELECT * from ck_gemm_tflops where Branch_name="develop" and timestamp = (SELECT MAX(timestamp));'''
        tflops_base = pd.read_sql_query(query, conn)

        #write new results to the db
        column_names=sorted_tests.insert(0,"Timestamp")
        column_names=column_names.insert(0,"Branch_name")
        values=sorted_tflops.insert(0,datetime.datetime.now())
        values=values.insert(0,branch_name)
        query='''INSERT INTO ck_gemm_tflops (column_names) VALUES(values);'''
        pd.read_sql_query(query, conn)
        conn.close()
    #compare the results to the baseline
    regression=0
    for i in len(tflops_base):
        if tflops_base[i]>1.1*sorted_tflops[i]:
            print("test # ",i,"shows regression")
            regression=1

    #return 0 if performance criteria met, otherwise return 1

    return regression

if __name__ == '__main__':
    main()