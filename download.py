"""
Modification of https://github.com/stanfordnlp/treelstm/blob/master/scripts/download.py

Downloads the following:
- Celeb-A dataset
"""

from __future__ import print_function
import urllib2
import sys
import os
import shutil
import zipfile
import gzip

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    u = urllib2.urlopen(url)
    f = open(filepath, 'wb')
    filesize = int(u.info().getheaders("Content-Length")[0])
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
            ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def download_celeb_a(dirpath):
    data_dir = 'celebA'
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Found Stanford POS Tagger - skip')
        return
    url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADVdnYbokd7TXhpvfWLL3sga/img_align_celeba.zip?dl=1'
    filepath = download(url, dirpath)
    zip_dir = ''
    with zipfile.ZipFile(filepath) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(filepath)
    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))

def download_celeb_a(dirpath):
    data_dir = 'celebA'
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Found Celeb-A - skip')
        return
    url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADVdnYbokd7TXhpvfWLL3sga/img_align_celeba.zip?dl=1'
    filepath = download(url, dirpath)
    zip_dir = ''
    with zipfile.ZipFile(filepath) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(filepath)
    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))

def _list_categories(tag):
    url = 'http://lsun.cs.princeton.edu/htbin/list.cgi?tag=' + tag
    f = urllib2.urlopen(url)
    return json.loads(f.read())

def _download_lsun(out_dir, category, set_name, tag):
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
          '&category={category}&set={set_name}'.format(**locals())
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)

def download_lsun(dirpath):
    data_dir = 'celebA'
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Found LSUN - skip')
        return

    tag = 'latest'
    categories = list_categories(tag)
    for category in categories:
        download(dirpath, category, 'train', tag)
        download(dirpath, category, 'val', tag)
    download(dirpath, '', 'test', tag)

if __name__ == '__main__':
    download_celeb_a('./data')
    download_lusn('./data')
