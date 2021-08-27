import datetime
import requests
import bs4
import os
import zlib

import geospacelab.toolbox.utilities.pydatetime as dttool

class Downloader(object):
    """
    A class to Download SSUSI data
    :param file_type:  "l1b", "edr-aur", or "sdr"
    """
    def __init__(self, dt_fr, dt_to, sat_id=None, orbit_id=None, data_file_root_dir=None, file_type='edr_aur'):

        dt_fr = dttool.get_start_of_the_day(dt_fr)
        dt_to = dttool.get_start_of_the_day(dt_to)
        self.file_type = file_type
        if dt_fr == dt_to:
            dt_to = dt_to + datetime.timedelta(hours=23, minutes=59)
        self.dt_fr = dt_fr  # datetime from
        self.dt_to = dt_to  # datetime to

        if data_file_root_dir is None:
            self.data_file_root_dir = pfr.datahub_data_root_dir / 'Madrigal' / 'GNSS' / 'TEC'
        else:
            self.data_file_root_dir = data_file_root_dir
        self.done = False

        self.madrigal_url = "http://cedar.openmadrigal.org/"
        self.download_madrigal_files()

    def __init__(self, ):
        """
        Set up some constants.
        Used for fitting.
        """
        self.baseUrl = "https://ssusi.jhuapl.edu/"
        self.outBaseDir = outBaseDir

    def download_files(self, inpDate, dataTypeList, satNum, orbitNum):
        """
        Get a list of the urls from input date
        and datatype and download the files
        and also move them to the corresponding
        folders.!!!
        """
        noData = True

        # construct day of year from date
        inpDoY = inpDate.timetuple().tm_yday
        inpDoY = "%03d" % inpDoY
        strDoY = str(inpDoY)
        #        if inpDoY < 10:
        #            strDoY = "00" + str(inpDoY)
        #        if ( inpDoY > 10) & (inpDoY < 100):
        #            strDoY = "0" + str(inpDoY)

        # construct url to get the list of files for the day
        for dataType in dataTypeList:
            payload = {"spc": satNum, "type": dataType, \
                       "Doy": strDoY, "year": str(inpDate.year)}
            # get a list of the files from dmsp ssusi website
            # based on the data type and date
            r = requests.get(self.baseUrl + "data_retriver/", \
                             params=payload, verify=False)
            soup = bs4.BeautifulSoup(r.text, 'html.parser')
            divFList = soup.find("div", {"id": "filelist"})
            hrefList = divFList.find_all(href=True)
            urlList = [self.baseUrl + hL["href"] for hL in hrefList]

            for fUrl in urlList:

                # we only need data files which have .NC
                if ".NC" not in fUrl:
                    continue
                # If working with sdr data use only
                # sdr-disk files
                if dataType == "sdr":
                    if "SDR-DISK" not in fUrl:
                        continue

                if len(orbitNum) == 5:
                    strOrbit = orbitNum
                    if strOrbit not in fUrl:
                        continue
                noData = False
                print("currently downloading-->", fUrl)
                rf = requests.get(fUrl, verify=False)
                currFName = rf.url.split("/")[-1]
                outDir = self.outBaseDir
                if not os.path.exists(outDir):
                    os.makedirs(outDir)
                with open(outDir + currFName, "wb") as ssusiData:
                    ssusiData.write(rf.content)
                self.filepath = outDir
                self.filename = currFName
            if noData:
                print("Warning: no available data!")


if __name__ == "__main__":
    ssObj = Downloader()
    inpDate = datetime.datetime(2015, 12, 5)
    dataTypeList = ["sdr"]  # , "l1b", "edr-aur" ]
    satNum = "f17"
    orbitNum = "46871"
    ssObj.download_files(inpDate, dataTypeList, satNum, orbitNum)