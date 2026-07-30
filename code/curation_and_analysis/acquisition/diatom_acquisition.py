"""
Diatom Database Acquisition
===========================

Original web-scraping pipeline used to assemble the Core Dataset of 12,353
diatom images from six public databases (ADIAC, AFD, DIA, DONA, FCE LTER,
LOIR) reported in:

    Yurtsever, U. (2026). Cross-Domain Evaluation of Modern Deep Learning
    Architectures for Microscopic Diatom Classification. (submitted)

Provided for methodological transparency and provenance, not as a maintained
or actively supported scraping tool. 

License: MIT (see LICENSE in the repository root)
Author : Ulaş Yurtsever, Sakarya University (ulas@sakarya.edu.tr)
"""

from bs4 import BeautifulSoup
import requests
import os
import json
from datetime import datetime
import shutil

import numpy as np
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage.transform import resize


class DiatomDatabase:
    """Per-source diatom image scraper with simple deduplication helpers.

    Each public-method (ADIAC, AFD, DIA, DONA, FCE_LTER, LOIR) targets a
    single upstream database and is intended to be invoked once. URLs and
    HTML structures reflect the state of each source at data-collection
    time (2023 Q3). Several endpoints have since changed or moved; see the
    accompanying README.md.
    """

    HEADERS = {
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/webp,image/apng,*/*;q=0.8"
        ),
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
        ),
    }

    REQUEST_TIMEOUT = 30  # seconds

    def __init__(self, path):
        """Initialise the scraper with an output root directory.

        Parameters
        ----------
        path : str
            Local directory under which per-database sub-folders will be
            created. Trailing slash is stripped.
        """
        self.path = path.rstrip("/")
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    # ------------------------------------------------------------------ #
    # Utilities                                                          #
    # ------------------------------------------------------------------ #

    def UniqueID(self):
        """Return a numeric timestamp string usable as a file-name suffix."""
        return str(datetime.timestamp(datetime.today())).replace(".", "")

    def DownloadImage(self, url, taxon, iscontinue, folder=""):
        """Fetch one image and store it under ``<path>/<folder>/<taxon>/``.

        Parameters
        ----------
        url : str
            Source image URL.
        taxon : str
            Genus or species name used as the leaf directory.
        iscontinue : bool
            If True and the file already exists, skip the download and
            return True; if False, the existing file is kept and a new
            file is downloaded with a timestamped suffix.
        folder : str, optional
            Per-database folder name (e.g. ``ADIAC_Database``).

        Returns
        -------
        bool
            ``True`` if the file was already present (and therefore
            skipped), ``False`` otherwise.
        """
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        if folder and not os.path.isdir(os.path.join(self.path, folder)):
            os.makedirs(os.path.join(self.path, folder))

        new_path = os.path.join(self.path, folder, taxon) if folder else os.path.join(self.path, taxon)
        if not os.path.isdir(new_path):
            os.makedirs(new_path)

        image_name = url.split("/")[-1]
        target = os.path.join(new_path, image_name)
        if os.path.isfile(target) and iscontinue:
            return True

        # If the file already exists, suffix with a unique id to avoid clobbering.
        if os.path.isfile(target):
            stem, _, ext = image_name.rpartition(".")
            image_name = f"{stem}_{self.UniqueID()}.{ext}"
            target = os.path.join(new_path, image_name)

        r = requests.get(url, headers=self.HEADERS, timeout=self.REQUEST_TIMEOUT)
        with open(target, "wb") as fh:
            fh.write(r.content)
        return False

    def Parsing(self, url, parser="lxml", data=None, req_type="post"):
        """Fetch ``url`` and return a parsed BeautifulSoup tree."""
        if req_type == "post":
            r = requests.post(url, data=data, headers=self.HEADERS, timeout=self.REQUEST_TIMEOUT).text
        elif req_type == "get":
            r = requests.get(url, data=data, headers=self.HEADERS, timeout=self.REQUEST_TIMEOUT).text
        else:
            raise ValueError(f"Unsupported req_type: {req_type!r}")
        return BeautifulSoup(r, parser)

    def CompareImages(self, img1_path, img2_path):
        """Return a near-identity score for two image files.

        The score is MSE-norm + SSIM. For pixel-identical images the
        return value is ``1.0`` (MSE = 0, SSIM = 1). Wrapped in a broad
        try/except so that unreadable files do not abort batch runs.
        """
        try:
            imga = io.imread(img1_path, as_gray=True)
            imgb = io.imread(img2_path, as_gray=True)

            imga_resize = resize(imga, (imga.shape[0], imga.shape[1]))
            imgb_resize = resize(imgb, (imga.shape[0], imga.shape[1]))

            img1 = img_as_float(rgb2gray(imga_resize))
            img2 = img_as_float(rgb2gray(imgb_resize))

            mse_value = np.linalg.norm(img1 - img2)
            ssim_value = ssim(img1, img2, data_range=img2.max() - img2.min())

            return mse_value + ssim_value
        except Exception:
            return 0

    def RemoveDoubleImages(self, exclude=None):
        """Walk ``self.path`` and delete byte-identical, pixel-identical pairs."""
        exclude = exclude or []
        count = 0
        start_date = datetime.now()
        d1 = sorted(os.listdir(self.path))
        print("The process was started on " + str(start_date))
        for sub1 in d1:
            sub1_path = os.path.join(self.path, sub1)
            if not (os.path.isdir(sub1_path) and sub1 not in exclude):
                continue
            for sub2 in sorted(os.listdir(sub1_path)):
                sub2_path = os.path.join(sub1_path, sub2)
                if not os.path.isdir(sub2_path):
                    continue
                img_list = [os.path.join(sub2_path, f) for f in sorted(os.listdir(sub2_path))]
                for i in range(len(img_list)):
                    for j in range(i + 1, len(img_list)):
                        if os.path.isfile(img_list[i]) and os.path.isfile(img_list[j]):
                            if os.path.getsize(img_list[i]) == os.path.getsize(img_list[j]):
                                if self.CompareImages(img_list[i], img_list[j]) == 1.0:
                                    os.remove(img_list[j])
                                    print("Removed File: " + img_list[j].replace(self.path, ""))
                                    count += 1
        print("----------oo----------")
        print(f"{start_date} - {datetime.now()} : "
              f"{int((datetime.now() - start_date).total_seconds())} second")
        print(f"{count} files were deleted...")

    def PrintDoubleAllImages(self, exclude=None):
        """List byte- and pixel-identical pairs across the whole tree (dry run)."""
        exclude = exclude or []
        count = 1
        img_list = []
        start_date = datetime.now()
        d1 = sorted(os.listdir(self.path))
        print("The process, which will take approximately 56 minutes, was started on "
              + str(start_date).split(".")[0])
        for sub1 in d1:
            sub1_path = os.path.join(self.path, sub1)
            if not (os.path.isdir(sub1_path) and sub1 not in exclude):
                continue
            for sub2 in sorted(os.listdir(sub1_path)):
                sub2_path = os.path.join(sub1_path, sub2)
                if not os.path.isdir(sub2_path):
                    continue
                for sub3 in sorted(os.listdir(sub2_path)):
                    img_list.append(os.path.join(sub2_path, sub3))
        for i in range(len(img_list)):
            for j in range(i + 1, len(img_list)):
                if os.path.getsize(img_list[i]) == os.path.getsize(img_list[j]):
                    if self.CompareImages(img_list[i], img_list[j]) == 1.0:
                        print(f"{count}: {img_list[i].replace(self.path, '')} "
                              f"--oo-- {img_list[j].replace(self.path, '')}")
                        count += 1
        print("----------oo----------")
        print(f"{start_date} - {datetime.now()} : "
              f"{int((datetime.now() - start_date).total_seconds())} second")
        print(f"{count} files that are identical were found.")

    # ------------------------------------------------------------------ #
    # Per-database scrapers                                              #
    # ------------------------------------------------------------------ #

    def AFD(self, folder="AFD_Database", iscontinue=False):
        """Antarctic Freshwater Diatoms (AFD) — McMurdo Dry Valleys LTER.

        Source: http://huey.colorado.edu/diatoms/
        """
        url = "http://huey.colorado.edu/diatoms/taxa/index.php"
        soup = self.Parsing(url)

        for sub1 in soup.findAll("div", attrs={"id": "onecol"}):
            for sub11 in sub1.findAll("table", attrs={"class": "tablesorter"}):
                for sub12 in sub11.findAll("tbody"):
                    for sub13 in sub12.findAll("td"):
                        for sub14 in sub13.findAll("a"):
                            taxon = ""
                            url2 = "http://huey.colorado.edu/diatoms/taxa/" + sub14["href"]
                            soup2 = self.Parsing(url2)

                            for sub2 in soup2.findAll("div", attrs={"id": "pagetitle"}):
                                for sub21 in sub2.findAll("h2", attrs={"class": "italicize"}):
                                    taxon = sub21.text.split(" ")[0]

                            for sub2 in soup2.findAll("div", attrs={"id": "onecol"}):
                                for sub21 in sub2.findAll("img"):
                                    image_file_url = (
                                        "http://huey.colorado.edu/diatoms/images_diatom_lg/"
                                        + sub21["src"].split("/")[-1]
                                    )
                                    iscontinue = self.DownloadImage(image_file_url, taxon, iscontinue, folder)
                                    print(taxon + ": " + image_file_url)
        print("AFD database is downloaded...")

    def DONA(self, folder="DONA_Database", iscontinue=False, morphological_group="centric"):
        """Diatoms of North America (DONA).

        Source: https://diatoms.org/

        Parameters
        ----------
        morphological_group : {"centric", "araphid", "raphid"}
            The original Core Dataset was assembled by invoking this method
            three times (once per morphological group) and concatenating the
            outputs. The default value preserves the original invocation;
            change it explicitly to scrape araphid or raphid genera.
        """
        url = "https://diatoms.org/actions/diatoms/ajax/getGenera"
        data = {"genusMorphologicalGroupSlug": morphological_group}
        soup = self.Parsing(url, data=data, req_type="get")

        soup_json = json.loads(soup.text)

        for sub1 in soup_json["groups"]:
            taxon = sub1["title"]
            for sub11 in sub1["species"]:
                url2 = sub11["url"]
                soup2 = self.Parsing(url2, req_type="get")
                for sub2 in soup2.findAll("div", attrs={"image-set"}):
                    for sub21 in sub2.findAll("a"):
                        image_file_url = sub21.find("img")["src"]
                        iscontinue = self.DownloadImage(image_file_url, taxon, iscontinue, folder)
                        print(taxon + ": " + image_file_url)
        print(f"DONA database ({morphological_group}) is downloaded...")

    def FCE_LTER(self, folder="FCE_LTER_Database", iscontinue=False):
        """Florida Coastal Everglades LTER Diatom Image Database (FCE LTER).

        Source: https://fce-lter.fiu.edu/data/database/diatom/
        """
        url = "https://fce-lter.fiu.edu/data/database/diatom/index.php?form_submitted=all_species"
        soup = self.Parsing(url)

        for sub1 in soup.find("main").parent.find_all("div"):
            for sub11 in sub1.findAll("div", attrs={"class": "row"}):
                for sub12 in sub11.findAll("div", attrs={"class": "small-12 columns text-left"}):
                    for sub13 in sub12.findAll("div", attrs={"style": "margin-left:20px;"}):
                        taxon = sub13.text.split(" ")[0]
                        url2 = "https://fce-lter.fiu.edu/data/database/diatom/" + sub13.find("a")["href"]
                        soup2 = self.Parsing(url2)
                        for sub2 in soup2.find("main").parent.find_all("div"):
                            for sub21 in sub2.findAll("article", attrs={"aria-label": "Search results"}):
                                for sub22 in sub21.findAll("div", attrs={"class": "row"}):
                                    for sub23 in sub22.findAll("div", attrs={"class": "small-12 medium-6 columns"}):
                                        for sub24 in sub23.findAll("img"):
                                            if len(sub24["src"]) > 7:
                                                image_file_url = (
                                                    "https://fce-lter.fiu.edu/data/database/diatom/"
                                                    + sub24["src"]
                                                )
                                                iscontinue = self.DownloadImage(image_file_url, taxon, iscontinue, folder)
                                                print(taxon + ": " + image_file_url)
        print("FCE_LTER database is downloaded...")

    def LOIR(self, folder="LOIR_Database", iscontinue=False):
        """DiatomLoir — marine and freshwater diatoms (M. Loir).

        Sources:
            http://www.diatomloir.eu/Diatodouces/
            http://www.diatomloir.eu/Site%20Diatom/
        """
        sites = [
            "http://www.diatomloir.eu/Diatodouces/",
            "http://www.diatomloir.eu/Site%20Diatom/",
        ]

        for url in sites:
            soup = self.Parsing(url, req_type="get")
            for sub1 in soup.findAll("a"):
                if sub1["href"].split("/")[-1] == "" and sub1["href"] != "/":
                    url2 = url + sub1["href"]
                    soup2 = self.Parsing(url2, req_type="get")
                    for sub2 in soup2.findAll("a"):
                        if sub2["href"].split(".")[-1] == "jpg":
                            taxon = sub2["href"].split(".")[0].split("%20")[0].strip().title()
                            image_file_url = url2 + sub2["href"]
                            iscontinue = self.DownloadImage(image_file_url, taxon, iscontinue, folder)
                            print(taxon + ": " + image_file_url)
        print("LOIR database is downloaded...")

    def DIA(self, folder="DIA_Database", iscontinue=False):
        """Diatom Image Archive (DIA) — Wunsam & Bowman, University of Alberta.

        Source: http://www.math.ualberta.ca/~bowman/diatom/
        """
        url = "http://www.math.ualberta.ca/~bowman/diatom/database/taxa/.index.js"
        soup = self.Parsing(url, req_type="get").text
        soup = soup[soup.find("(") + 2: soup.find(");")]
        soup = soup.replace("\t", "").replace("\n", "").replace("'", "").split(",")

        for sub1 in soup:
            url2 = ("http://www.math.ualberta.ca/~bowman/diatom/database/taxa/"
                    + sub1 + "/.index.js")

            # Diatom images on the right side of the web page.
            soup2 = self.Parsing(url2, req_type="get").text
            soup2 = soup2[
                soup2.find("var imageNames = new Array(") + 28:
                soup2.rfind("// thumbnail coordinates (imageThumbCoords[page number])") - 5
            ]
            soup2 = soup2.replace("new Array(", "").replace(")", "")
            soup2 = soup2.replace("\t", "").replace("\n", "").replace("'", "").split(",")
            for sub11 in soup2:
                if sub11 != "":
                    image_file_url = ("http://www.math.ualberta.ca/~bowman/diatom/database/taxa/"
                                      + sub1 + "/" + sub11)
                    taxon = sub1
                    iscontinue = self.DownloadImage(image_file_url, taxon, iscontinue, folder)
                    print(taxon + ": " + image_file_url)

            # Diatom images on sub-web pages.
            soup2 = self.Parsing(url2, req_type="get").text
            soup2 = soup2[soup2.find("(") + 2: soup2.find(");")]
            soup2 = soup2.replace("\t", "").replace("\n", "").replace("'", "").split(",")
            for sub11 in soup2:
                url3 = ("http://www.math.ualberta.ca/~bowman/diatom/database/taxa/"
                        + sub1 + "/" + sub11 + "/.index.js")
                soup3 = self.Parsing(url3, req_type="get").text
                soup3 = soup3[
                    soup3.find("var imageNames = new Array(") + 28:
                    soup3.rfind("// thumbnail coordinates (imageThumbCoords[page number])") - 5
                ]
                soup3 = soup3.replace("new Array(", "").replace(")", "")
                soup3 = soup3.replace("\t", "").replace("\n", "").replace("'", "").split(",")
                for sub12 in soup3:
                    if sub12 != "":
                        image_file_url = ("http://www.math.ualberta.ca/~bowman/diatom/database/taxa/"
                                          + sub1 + "/" + sub11 + "/" + sub12)
                        taxon = sub1
                        iscontinue = self.DownloadImage(image_file_url, taxon, iscontinue, folder)
                        print(taxon + ": " + image_file_url)

        print("DIA database is downloaded...")

    def ADIAC(self, folder="ADIAC_Database", iscontinue=False):
        """ADIAC — Automatic Diatom Identification And Classification.

        Source: https://rbg-web2.rbge.org.uk/ADIAC/

        Bulk-downloads the public ``tar.gz`` archives published by the
        ADIAC project at the Royal Botanic Garden Edinburgh, extracts the
        ``.TIF`` images, and reorganises them by taxon name using the
        accompanying ``index.csv`` and a fixed list of letter suffixes.
        """
        url = "https://rbg-web2.rbge.org.uk/ADIAC/pubdat/downloads/public_images.htm"
        soup = self.Parsing(url)

        download_file = []
        for sub in soup.findAll("ul"):
            for sub1 in sub.findAll("li"):
                file_url = ("https://rbg-web2.rbge.org.uk/ADIAC/pubdat/downloads/"
                            + sub1.find("a")["href"])
                download_file.append(file_url)

        # Download the archive files.
        for i, file_url in enumerate(download_file):
            iscontinue = self.DownloadImage(file_url, "_Temp", iscontinue, folder)
            print(f"Downloaded ({i + 1}/{len(download_file)}): {file_url}")

        # Extract every tar.gz in the temp directory.
        temp_root = os.path.join(self.path, folder, "_Temp")
        files = sorted(os.listdir(temp_root))
        for fname in files:
            if fname.split(".")[-1] == "gz":
                file_name = os.path.join(temp_root, fname)
                shutil.unpack_archive(file_name, temp_root)
                print("Extracted : " + file_name)

        # Move all TIF images out of nested archive directories into one folder.
        directories = sorted(os.listdir(temp_root))
        directories = list(set(directories) - set(files))
        for directory in directories:
            images = os.listdir(os.path.join(temp_root, directory))
            for img in images:
                if img.split(".")[-1] in ("TIF",):
                    try:
                        shutil.move(os.path.join(temp_root, directory, img),
                                    os.path.join(self.path, folder))
                    except Exception:
                        os.remove(os.path.join(temp_root, directory, img))
                        continue

        # Remove the (now empty) per-archive directories.
        for directory in directories:
            shutil.rmtree(os.path.join(temp_root, directory))

        # Re-bucket the flat .TIF list under per-taxon folders via index.csv.
        with open(os.path.join(temp_root, "index.csv"), "rb") as f:
            lines = [l.decode("utf8", "ignore") for l in f.readlines()]
            # ADIAC file-name suffixes: each base ID may carry zero or more
            # of these letter codes (e.g. "ASS0123", "ASS0123A", "ASS0123AB").
            suffixes = ["", "A", "B", "C", "D", "E", "AA", "AB", "AC", "AD", "AE",
                        "BA", "BB", "BC", "BD", "BE", "CA", "CB", "CC", "CD", "CE",
                        "DA", "DB", "DC", "DD", "DE", "EA", "EB", "EC", "ED", "EE"]
            new_path = os.path.join(self.path, folder) + "/"
            for i, row in enumerate(lines):
                if i > 0:
                    taxon = row.split(",")[1].split(" ")[0].replace('"', "")
                    for suffix in suffixes:
                        file2 = new_path + row.split(",")[0][0:6] + suffix + ".TIF"
                        if os.path.isfile(file2):
                            if not os.path.isdir(new_path + taxon):
                                os.mkdir(new_path + taxon)
                            shutil.move(file2, new_path + taxon)

        shutil.rmtree(temp_root)
        print("ADIAC database is downloaded...")
