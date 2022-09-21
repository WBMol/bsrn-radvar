from datetime import timedelta
import os
import pysftp
from ftplib import FTP
from datetime import datetime

from scripts.processor import settings


def download_1sec_bsrn_ftp(time_start, time_end):
    """
    function downloads data for the (soon to be removed) KNMI bbc ftp server

    :param datetime time_start:
    :param datetime time_end:
    :return:
    """
    # assign sftp variable so loop can check whether there is an existing connection already
    sftp = None

    # start the loop to find/download files
    time_iter = time_start
    while time_iter < time_end:
        # setup file name and paths
        file_name = time_iter.strftime("BSRN_%Y%m%d_%H%M.cdf.gz")
        file_path = settings.fdir_raw_1sec.format(y=time_iter.year, m=time_iter.month, d=time_iter.day)

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # download file (only when it does not yet exist locally, including unzipped version)
        check_1 = os.path.isfile(os.path.join(file_path, file_name))
        check_2 = os.path.isfile(os.path.join(file_path, file_name[:-3]))
        if check_1:
            print("Skip downloading {:}, file already exists".format(file_name))
        elif check_2:
            print("Skip downloading {:}, unzipped file already exists".format(file_name))
        else:
            # setup up host keys file
            cnopts = pysftp.CnOpts()
            cnopts.hostkeys = None  # risky! https://stackoverflow.com/questions/38939454/verify-host-key-with-pysftp

            # setup ftp connection
            if sftp is None:
                sftp = pysftp.Connection(settings.ftp_url, cnopts=cnopts,
                                         username=settings.ftp_user, password=settings.ftp_pass)

            # download file using ftp
            path_name = time_iter.strftime("MEMBERS/knmi/KNMI_ACQ/BSRN/%Y/%m/%d/")
            with sftp.cd(path_name):
                target_file_path = os.path.join(file_path, file_name)
                try:
                    sftp.get(file_name, target_file_path)
                except FileNotFoundError:
                    print('failed to download: %s (file not found)' % file_name)

            print("%s: downloaded to %s" % (file_name, file_path))

        time_iter += timedelta(minutes=10)


def download_1min_bsrn_ftp(dt, station='cab'):
    """
    Downloads monthly data from the official BSRN ftp
    more info: https://bsrn.awi.de/data/data-retrieval-via-ftp/

    :param datetime dt: the year and month to download
    :param str station: 3-letter name of the station
    :return:
    """

    # ftp settings
    url = 'ftp.bsrn.awi.de'
    usr = 'bsrnftp'
    pwd = 'bsrn1'

    # generate file path
    fdir_out = settings.fdir_raw_1min.format(res='1min', y=dt.year)
    fname = station + dt.strftime('%m%y.dat.gz')
    fpath = os.path.join(fdir_out, fname)

    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)
    if os.path.isfile(fpath):
        print("File already exists: %s" % fpath)
        return

    # download
    with FTP(url, usr, pwd) as ftp:
        ftp.cwd(station)
        with open(fpath, 'wb') as f:
            ftp.retrbinary('RETR ' + fname, f.write)
    print("Downloaded data for station '%s', %s" % (station, dt.strftime('%Y-%m')))


if __name__ == "__main__":
    for year in range(2006, 2021):
        for month in range(1, 13):
            download_1min_bsrn_ftp(dt=datetime(year, month, 1))
