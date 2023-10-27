import argparse
import pandas as pd
from time import perf_counter
import multiprocessing as mp
import os


class controller:
    def __init__(self, namespace):
        self.args = namespace
        self.args.data_dir = self.args.script_dir + '/' + self.args.data_dir

    def run(self):
        args_str = ''
        for key, value in vars(self.args).items():
            if key != 'verbose' and key != 'script_dir' and key != 'script_name':
                if isinstance(value, bool) and value:
                    args_str += ' --' + key

                if not isinstance(value, bool):
                    args_str += ' --' + key + ' ' + str(value)

        os.system('python ' + self.args.script_dir + '/' + self.args.script_name + args_str)
        # subprocess.call(self.args.script_dir + '/' + self.args.script_name + args_str)

def runner(namespace):
    job = controller(namespace)
    job.run()

# convert param df into namespace list for pool.map(), also append verbose flag
def param_lister(input_df, retxt, num_jobs):
    ns_list = []
    for i in range(num_jobs):
        flags = pd.Series([False, retxt], index=['verbose', 'retxt'])
        pre_ns = pd.concat([input_df.iloc[i, :], flags]).to_dict()
        ns_list.append(argparse.Namespace(**pre_ns))
    return ns_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='t-NN Experiment Scheduler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Descripti
    )
    parser.add_argument('filename', nargs='?', default='tmp.csv', help='filename/path of parameter CSV')
    parser.add_argument('-m', '--multi', dest='multi', action='store_true', help='enable multiprocessing')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='print output to console, forces single-processing!')
    args = parser.parse_args()

    input_df = pd.read_csv(args.filename, sep=',')
    num_jobs, _ = input_df.shape

    # -v overrides -m and forces single-processing
    if args.verbose:
        args.multi = False

    # no multiprocessing: do jobs sequentially
    if not args.multi:
        t_initial = perf_counter()

        # convert param df into namespace list, also append verbose flag
        for i in range(num_jobs):
            pre_ns = input_df.iloc[i, :].to_dict()
            pre_ns['verbose'] = args.verbose
            job = controller(argparse.Namespace(**pre_ns))
            job.run()

        t_final = perf_counter()
        print("Time elapsed: " + str(t_final - t_initial) + ' secs')

    # multiprocessing
    if args.multi:
        if mp.cpu_count() > 1:
            proc_size = mp.cpu_count() - 1  # max recruited processes
        else:
            proc_size = 1
        t_initial = perf_counter()

        param_list = param_lister(input_df, args.retxt, num_jobs)

        with mp.Pool(processes=proc_size) as pool:
            pool.map(runner, param_list)

        t_final = perf_counter()
        print("Time elapsed: " + str(t_final - t_initial) + ' secs')
