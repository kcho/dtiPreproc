#!/Users/admin/anaconda/bin/python

import textwrap
import dicom
import re
import os
import shutil
import sys
import argparse


# In[83]:

def getDTIdirectory(directory):
    '''
    Input : subject directory location
    Output : DTIdirectories
    '''
    DTIdirectories = []
    for root, dirs, files in os.walk(directory):

        #Looping through the bunch of files
        for directory in dirs:

            if re.search('DTI_72D',directory,flags=re.IGNORECASE):
                DTIdirectories.append(os.path.abspath(directory))

    return DTIdirectories

def dicomConversion(outputDir,DTIdirectories):
    # Make output directory
    try:
        os.mkdir(outputDir)
    except:
        pass

    # Dicom conversion
    if len(os.listdir(outputDir)) == 0: # if empty
        print '\tDicom Conversion'
        print '\t----------------'
        for DTIdirectory in DTIdirectories:
            command = '/ccnc_bin/mricron/dcm2nii -o {outputDir} \
                        {DTIdirectory}'.format(
                                            outputDir=outputDir,
                                            DTIdirectory=DTIdirectory)
            dcm2niiConversionOutput = os.popen(command).read()

def nameChange(outputDir):
    # Name change
    try:
        initialFiles = os.listdir(outputDir)
        bval = ''.join([x for x in initialFiles if re.search('bval$',x)])
        bvec = ''.join([x for x in initialFiles if re.search('bvec$',x)])
        data = ''.join([x for x in initialFiles if re.search('DTI.*AP.*nii.gz',x)])
        P2A_b0 = ''.join([x for x in initialFiles if re.search('DTI.*PA.*nii.gz',x)])

        shutil.move(os.path.join(outputDir,bval),os.path.join(outputDir,'bvals'))
        shutil.move(os.path.join(outputDir,bvec),os.path.join(outputDir,'bvecs'))
        shutil.move(os.path.join(outputDir,data),os.path.join(outputDir,'data.nii.gz'))
        shutil.move(os.path.join(outputDir,P2A_b0),os.path.join(outputDir,'P2A_b0.nii.gz'))
    except:
        pass
def extractB0images(outputDir):
    #Extract B0 images from the data
    print '\tExtract B0 images'
    print '\t----------------'
    if len([x for x in os.listdir(outputDir) if x.startswith('A2P_b0_')]) !=  9:
        b0Nums = [0,1,10,19,28,37,46,55,64]

        for b0Num in b0Nums:
            command = 'fslroi {outputDir}/data \
                    {outputDir}/A2P_b0_{0} \
                    {0} 1'.format(b0Num,
                    outputDir=outputDir)
            print os.popen(command).read()+'.',

    # Merge B0s
    if not os.path.isfile(os.path.join(
                            outputDir,
                            'b0_images.nii.gz')):
        command = 'fslmerge -t {outputDir}/b0_images \
                {outputDir}/*_b0*'.format(
                                outputDir=outputDir)
        print os.popen(command).read()

def writeAcqParams(outputDir):
    print '\tWrite Acquisition Parameters'
    print '\t----------------'
    if not os.path.isfile(os.path.join(outputDir,
        'acqparams.txt')):
        # Writing acqparams.txt
        acqparams = '''0 -1 0 0.0773
    0 -1 0 0.0773
    0 -1 0 0.0773
    0 -1 0 0.0773
    0 -1 0 0.0773
    0 -1 0 0.0773
    0 -1 0 0.0773
    0 -1 0 0.0773
    0 -1 0 0.0773
    0 1 0 0.0773
    0 1 0 0.0773
    0 1 0 0.0773
    0 1 0 0.0773
    0 1 0 0.0773
    0 1 0 0.0773
    0 1 0 0.0773
    0 1 0 0.0773'''
        with open(os.path.join(outputDir,
                               'acqparams.txt'),'w') as f:
            f.write(acqparams)

def makeEvenNumB0(outputDir):
    print '\tMake the slice number even'
    print '\t---------------------------'
    # Make the slice number even
    if not os.path.isfile(os.path.join(outputDir,
        'b0_images_even.nii.gz' )):
        # split B0
        command = 'fslslice {outputDir}/b0_images'.format(outputDir=outputDir)
        fslsliceOutput = os.popen(command).read()

        # merge B0
        slicedImages = [os.path.join(outputDir,x) for x in os.listdir(outputDir) if re.search('slice',x)]

        command = 'fslmerge -z \
                {outputDir}/b0_images_even \
                {slicedImages}'.format(
                outputDir=outputDir,
                slicedImages=' '.join(slicedImages[:-1]))
        fslmergeOutput = os.popen(command).read()

def topup(outputDir):
    if os.path.isfile(os.path.join(
        outputDir,
        'unwarped_images.nii.gz')):
        pass
    else:
        print '\tRunning Topup, FSL'
        print '\t---------------------------'
        command = 'topup --imain={outputDir}/b0_images_even \
                --datain={outputDir}/acqparams.txt \
                --config=b02b0.cnf \
                --out={outputDir}/topup_results \
                --fout={outputDir}/field \
                --iout={outputDir}/unwarped_images'.format(
                        outputDir=outputDir)

        print os.popen(command).read()

def main(args):

    ################################################
    # InputDir specification
    ################################################
    DTIdirectories = getDTIdirectory(args.directory)

    ################################################
    # OutputDir specification
    ################################################
    outputDir = os.path.join(args.directory,'DTIpreproc')

    ################################################
    # Preparation
    ################################################
    dicomConversion(outputDir,DTIdirectories)
    nameChange(outputDir)
    extractB0images(outputDir)
    writeAcqParams(outputDir)
    makeEvenNumB0(outputDir)

    ################################################
    # Running topup
    ################################################
    topup(outputDir)

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
            description = textwrap.dedent('''\
                    {codeName} : Pre-process the new diffusion tensor images
                    ==========================================================
                        eg) {codeName}
                        eg) {codeName} --dir /Users/kevin/NOR04_CKI
                        eg) {codeName} --dir /Users/kevin/NOR04_CKI
                    '''.format(codeName=os.path.basename(__file__))))

            #epilog="By Kevin, 26th May 2014")
    parser.add_argument('-dir','--directory',help='Data directory location, default = pwd',default=os.getcwd())
    args = parser.parse_args()
    main(args)

