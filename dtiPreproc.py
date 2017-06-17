#!/ccnc_bin/venv/bin/python
import textwrap
from os.path import join, basename, isfile, isdir, dirname
import dicom
import re
import os
import shutil
import sys
import argparse
import nibabel as nb
import numpy as np


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
                DTIdirectories.append(os.path.join(root,directory))

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
        print '\t--------------------------------'
        for DTIdirectory in DTIdirectories:
            command = '/ccnc_bin/mricron/dcm2nii -o {outputDir} \
                    "{DTIdirectory}"'.format(
                                            outputDir=os.path.abspath(outputDir),
                                            DTIdirectory=DTIdirectory)
            dcm2niiConversionOutput = os.popen(command).read()

def nameChange(outputDir):
    # Name change
    try:
        initialFiles = os.listdir(outputDir)
        bval = ''.join([x for x in initialFiles if re.search('bval$',x)][0])
        bvec = ''.join([x for x in initialFiles if re.search('bvec$',x)][0])
        data = ''.join([x for x in initialFiles if re.search('DTI.*AP.*nii.gz',x)][0])
        P2A_b0 = ''.join([x for x in initialFiles if re.search('DTI.*PA.*nii.gz',x)][0])

        print os.path.join(outputDir,bval),os.path.join(outputDir,'bvals')
        print os.path.join(outputDir,bvec),os.path.join(outputDir,'bvecs')
        print os.path.join(outputDir,data),os.path.join(outputDir,'data.nii.gz')
        print os.path.join(outputDir,P2A_b0),os.path.join(outputDir,'P2A_b0.nii.gz')

        shutil.move(os.path.join(outputDir,bval),os.path.join(outputDir,'bvals'))
        shutil.move(os.path.join(outputDir,bvec),os.path.join(outputDir,'bvecs'))
        shutil.move(os.path.join(outputDir,data),os.path.join(outputDir,'data.nii.gz'))
        shutil.move(os.path.join(outputDir,P2A_b0),os.path.join(outputDir,'P2A_b0.nii.gz'))
    except:
        pass

def makeEvenNumB0(niftiImg, outdir):
    outputDir = os.path.dirname(niftiImg)

    print '\tMake the slice number even : ', os.path.basename(niftiImg)
    print '\t--------------------------------'

    f = nb.load(niftiImg).get_data()
    sliceNum = f.shape[2]

    # Make the slice number even
    if not sliceNum % 2 == 0:
        print '\t\tRemoving the most inferior slice'
        # 3D
        try:
            newData = f[:,:,:-1]
        # 4D
        except:
            newData = f[:,:,:-1,:]

        # split B0
        command = 'fslslice {0}'.format(niftiImg)
        fslsliceOutput = os.popen(command).read()

        # merge B0
        slicedImages = [os.path.join(outputDir,x) for x in os.listdir(outputDir) if re.search('slice',x)]

        even_niftiImg = os.path.join(outdir, 
                                'even_' + os.path.basename(niftiImg))

        command = 'fslmerge -z \
                {newImg}\
                {slicedImages}'.format(
                    newImg = even_niftiImg,
                    slicedImages=' '.join(slicedImages[:-1]))

        fslmergeOutput = os.popen(command).read()

        #Remove splitImages
        for img in slicedImages:
            os.remove(img)

        return even_niftiImg
    else:
        return niftiImg

def extractB0images(niftiImg,bval, outDir):
    img = nb.load(niftiImg)
    imgData = img.get_data()

    if bval:
        with open(bval, 'r') as f:
            bvals = f.read().split(' ')

        bvalsArray = np.array(bvals)
        b0_indexArray = np.arange(len(bvalsArray))[bvalsArray=='0']

    else:
        try:
            b0_indexArray = range(imgData.shape[3])
        except:
            b0_indexArray = [0]

    #all_b0_imgData = np.zeros_like(imgData[:,:,:,0])
    outImgLocs = []
    for b0_num, b0_index in enumerate(b0_indexArray):
        try:
            b0_data = imgData[:,:,:,b0_index]
        except:
            b0_data = imgData
        newData = nb.Nifti1Image(b0_data, img.affine)
        imgName = 'b0_'+str(b0_num)+'_'+os.path.basename(niftiImg)
        outImgLoc = os.path.join(outDir, imgName)
        newData.to_filename(outImgLoc)
        outImgLocs.append(outImgLoc)

    return outImgLocs

def extractMeanB0images(niftiImg,bval, outDir):
    img = nb.load(niftiImg)
    imgData = img.get_data()

    if bval:
        with open(bval, 'r') as f:
            bvals = f.read().split(' ')

        bvalsArray = np.array(bvals)
        b0_indexArray = np.arange(len(bvalsArray))[bvalsArray=='0']

    else:
        try:
            b0_indexArray = range(imgData.shape[3])
        except:
            b0_indexArray = [0]

    all_b0_imgData = np.zeros_like(imgData[:,:,:,0])
    for b0_index in b0_indexArray:
        b0_data = imgData[:,:,:,b0_index]
        all_b0_imgData = all_b0_imgData + b0_data

    mean_b0_image = all_b0_imgData / len(b0_indexArray)
    newData = nb.Nifti1Image(mean_b0_image, img.affine)

    imgName = os.path.basename(niftiImg)
    outImgLoc = os.path.join(outDir, imgName)
    newData.to_filename(outImgLoc)

    return outImgLoc
    

#def extractB0images():

    ##if old:
        ##command = 'fslroi {outputDir}/data.nii.gz \
                ##{outputDir}/nodif.nii.gz \
                ##{0} 1'.format(b0Num,
                ##outputDir=outputDir)
        ##os.popen(command).read()

    ##else:
        ## full version
        ## extract the b0 images that were taken between the diffusion weighted images
        #if full and len([x for x in os.listdir(outputDir) if x.startswith('A2P_b0_')]) != 9:
            #b0Nums = [0,1,10,19,28,37,46,55,64]

            ## for all the sequences of the b0 images
            #for b0Num in b0Nums:
                #command = 'fslroi {outputDir}/data_even \
                        #{outputDir}/A2P_b0_{0} \
                        #{0} 1'.format(b0Num,
                        #outputDir=outputDir)
                #os.popen(command).read()

        ## non full version
        #elif not full and len([x for x in os.listdir(outputDir) if x.startswith('A2P_b0_')]) != 2:
            #b0Nums = [0,1] # Two B0s from A >> P
            #for b0Num in b0Nums:
                #command = 'fslroi {outputDir}/data_even \
                        #{outputDir}/A2P_b0_{0} \
                        #{0} 1'.format(b0Num,
                        #outputDir=outputDir)
                #os.popen(command).read()

        ## Merge B0s extracted from AP
        #if not os.path.isfile(os.path.join(
                                #outputDir,
                                #'b0_images.nii.gz')):
            #if full:
                #command = 'fslmerge -t {outputDir}/b0_images \
                        #{outputDir}/*_b0*'.format(
                                        #outputDir=outputDir)
                #os.popen(command).read()
            #else:
                ## Two images of P2A b0
                #b0Nums = [0,1] # Two B0s from A >> P
                #for b0Num in b0Nums:
                    #command = 'fslroi {outputDir}/P2A_b0_even \
                            #{outputDir}/P2A_b0_{0} \
                            #{0} 1'.format(b0Num,
                            #outputDir=outputDir)
                    #os.popen(command).read()

                ##merge above two mean images
                #command = 'fslmerge -t {outputDir}/b0_images \
                        #{outputDir}/[PA]2[AP]_b0_[[:digit:]]*.nii.gz'.format(
                                        #outputDir=outputDir)
                #fslmathsOutput = os.popen(command).read()


def writeAcqParams(ap_b0_num, pa_b0_num, outDir,full):
    '''
    A >> P : 0 -1 0
    A << P : 0 1 0
    4th number : 0.69 ms * 112 * 0.001
    https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=ind1306&L=fsl&D=0&P=49368
    > TR = 9300ms , TE=94ms, Echo spacing = 0.69ms, 96x96 matrix and 65 slices,
    > Phase partial Fourier 6/8 and finally bandwidth 1628Hz/Px
    the relevant time in your case is (96-1)*0.00069 = 0.0656 seconds.

    For SCS project
    - 112x112 matrix
    - 0.8 echo spacing
    (112-1) * 0.0008 = .0888
    '''

    print '\tWrite Acquisition Parameters'
    print '\t--------------------------------'
    if not os.path.isfile(os.path.join(outDir,
        'acqparams.txt')):
        # Writing acqparams.txt
        if full:
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
        else:
            #Temple
            #acqparams = '''0 -1 0 0.0773
#0 -1 0 0.0773
#0 1 0 0.0773
#0 1 0 0.0773'''
            acqparams = '0 -1 0 0.0888\n'*ap_b0_num+ '0 1 0 0.0888\n'*pa_b0_num

        with open(os.path.join(outDir,
                               'acqparams.txt'),'w') as f:
            f.write(acqparams)

    return os.path.join(outDir,'acqparams.txt')


def topup(merged_b0_all, acqparam, outDir):
    print '\tRunning Topup, FSL'
    print '\t--------------------------------'
    if os.path.isfile(os.path.join(
        outDir,
        'unwarped_images.nii.gz')):
        pass
    else:
        command = 'topup --imain={b0_images} \
                --datain={acq} \
                --config=b02b0.cnf \
                --out={outDir}/topup_results \
                --fout={outDir}/field \
                --iout={outDir}/unwarped_images'.format(
                        b0_images = merged_b0_all,
                        acq = acqparam, 
                        outDir=outDir)

        output=os.popen(command).read()


def applytopup(ap_b0, pa_b0, acqparam, outDir):
    print '\tApply Topup'
    print '\t--------------------------------'

    # inindex number
    # index number of the ap_b0 / pa_b0 image 
    # in the acqparam
    with open(acqparam, 'r') as f:
        lines = f.readlines()
    lines_strip = [x.strip() for x in lines]
    ap_b0_num = lines_strip.index(list(set(lines_strip))[0]) + 1 
    pa_b0_num = lines_strip.index(list(set(lines_strip))[1]) + 1

    if os.path.isfile(os.path.join(
        outDir,
        'data_topup.nii.gz')):
        pass
    else:
        command = 'applytopup \
                --imain={ap_b0},{pa_b0} \
                --datain={acq} \
                --inindex={ap_b0_num},{pa_b0_num} \
                --topup={outDir}/topup_results \
                --out={outDir}/hifi_nodif.nii.gz \
                --method=jac'.format(outDir=outDir,
                        ap_b0 = ap_b0,
                        pa_b0 = pa_b0,
                        ap_b0_num = ap_b0_num,
                        pa_b0_num = pa_b0_num,
                        acq = acqparam)
        applyTopUpOutput = os.popen(command).read()

def eddy(ap_nifti_even, bvals, bvecs, acqparam, outDir):
    print '\tEddy Correction'
    print '\t--------------------------------'
    if os.path.isfile(os.path.join(
        outDir,
        'eddy_unwarped_images.nii.gz')):

        bet_mask = os.path.join(outDir,
                                'hifi_nodif_brain_mask.nii.gz')
        eddy_out = os.path.join(outDir, 
                                'eddy_unwarped_images.nii.gz')
    else:
        ## mean of the corrected image
        #mean(os.path.join(outDir,'b0_images'),
                #os.path.join(outDir,'b0_images_mean'))

        # bet
        bet_in = os.path.join(outDir,'hifi_nodif')
        bet_out = os.path.join(outDir,'hifi_nodif_brain')
        bet_mask = os.path.join(outDir,
                                'hifi_nodif_brain_mask.nii.gz')
        os.system('bet {inImg} {output} \
                -c 54 56 32 \
                -m -f 0.25'.format(inImg=bet_in, output=bet_out))
        #os.system('bet {inImg} {output} -m'.format(
            #inImg = os.path.join(outDir,'b0_images_mean'),
            #output = os.path.join(outDir,'b0_images_mean_brain')))

        # index
        with open(bvals,'r') as f:
            line = f.read()
        vol_num = len(line.strip().split(' '))

        # create an index file
        # 70 --> number of volumes SCS project
        index = ['1']*vol_num
        index = ' '.join(index)

        with open(os.path.join(outDir,'index.txt'),'w') as f:
            f.write(index)

        #eddy
        command = 'eddy \
                --imain={ap} \
                --mask={outDir}/hifi_nodif_brain_mask \
                --acqp={acq} \
                --index={outDir}/index.txt \
                --bvecs={bvecs} \
                --bvals={bvals} \
                --fwhm=0 \
                --flm=quadratic \
                --topup={outDir}/topup_results \
                --out={outDir}/eddy_unwarped_images'.format(
                        ap = ap_nifti_even,
                        acq = acqparam,
                        bvecs = bvecs,
                        bvals = bvals,
                        outDir=outDir)
        eddyOutput = os.popen(command).read()
        eddy_out = os.path.join(outDir, 
                                'eddy_unwarped_images.nii.gz')
        print eddyOutput

    try:
        return eddy_out, bet_mask
    except:
        return eddy_out, bet_mask


def mean(srcImg,trgImg):
    os.system('fslmaths {src} -Tmean {out}'.format(
        src=srcImg,
        out=trgImg))

def dtifit(eddy_out, mask, bvecs, bvals, outName, outDir):
    print '\tDTIFIT : scalar map calculation'
    print '\t--------------------------------'
    command = 'dtifit \
            -k {eddy_out} \
            -m {mask} \
            -r {bvecs} \
            -b {bvals} \
            -o {outDir}/{outName}'.format(outDir=outDir,
                    eddy_out=eddy_out,
                    mask=mask,
                    bvecs=bvecs,
                    bvals=bvals,
                    outName=outName)
    print os.popen(command).read()



def dtiPreproc(ap_nifti, ap_bvec, ap_bval, pa_nifti, outDir):
    # b0 images registration required
    # b0 images registration required
    # b0 images registration required
    # b0 images registration required
    # b0 images registration required
    ap_nifti_even = makeEvenNumB0(ap_nifti, outDir)
    pa_nifti_even = makeEvenNumB0(pa_nifti, outDir)

    ap_b0_list = extractB0images(ap_nifti, ap_bval, outDir)
    pa_b0_list = extractB0images(pa_nifti, False, outDir)

    # Merge ap b0 images
    ap_b0_data_list = []
    for ap_b0 in ap_b0_list:
        ap_b0_data = nb.load(ap_b0).get_data()
        ap_b0_data_list.append(ap_b0_data)

    ap_b0_all_data = np.concatenate([x[...,np.newaxis] for x in ap_b0_data_list],
            axis=3)

    # Merge pa b0 images
    pa_b0_data_list = []
    for pa_b0 in pa_b0_list:
        pa_b0_data = nb.load(pa_b0).get_data()
        pa_b0_data_list.append(pa_b0_data)

    pa_b0_all_data = np.concatenate([x[...,np.newaxis] for x in pa_b0_data_list],
            axis=3)

    # Merge ap & pa b0 images
    merged_b0_all_data = np.concatenate(
            (ap_b0_all_data[...,np.newaxis],
             pa_b0_all_data[...,np.newaxis]),axis=3)
    merged_b0_all = os.path.join(outDir,'merged_b0.nii.gz')
    f = nb.load(ap_b0_list[0])
    nb.Nifti1Image(merged_b0_all_data, 
                   f.affine).to_filename(merged_b0_all)
    
    acqparam = writeAcqParams(len(ap_b0_list),
            len(pa_b0_list),
            outDir,False)
    topup(merged_b0_all, acqparam, outDir)
    applytopup(ap_b0_list[0], pa_b0_list[0], acqparam, outDir)
    eddy_out, mask = eddy(ap_nifti_even, ap_bval, ap_bvec, acqparam, outDir)
    dtifit(eddy_out, mask, ap_bvec, ap_bval, 'dti', outDir)


def get_dti_trio(Loc):
    for root, dirs, files in os.walk(Loc):
        for f in files:
            if 'nii' in f:
                nifti = os.path.join(root, f)
            if 'bvec' in f:
                bvec = os.path.join(root, f)
            if 'bval' in f:
                bval = os.path.join(root, f)
    return nifti, bvec, bval

def get_nifti(Loc):
    for root, dirs, files in os.walk(Loc):
        for f in files:
            if 'nii' in f:
                nifti = os.path.join(root, f)
                return nifti


if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
            description = textwrap.dedent('''\
                    {codeName} : Pre-process the new diffusion tensor images
                    ==========================================================
                        eg) {codeName}
                        eg) {codeName} --dir /Users/kevin/NOR04_CKI
                        eg) {codeName} --dir /Users/kevin/NOR04_CKI
                    '''.format(codeName=os.path.basename(__file__))))
    parser.add_argument('-d','--dir',help='Data directory location', default=os.getcwd())
    parser.add_argument('-a','--apDir',help='AP data directory location', default=join(os.getcwd(), 'DTI_AP'))
    parser.add_argument('-p','--paDir',help='PA data directory location', default=join(os.getcwd(), 'DTI_PA'))
    parser.add_argument('-f','--full',help='Process all B0', default = False)
    parser.add_argument('-t','--dtifit',help='Create FA maps', default = True)
    parser.add_argument('-o','--old',help='Short version', default = False)
    parser.add_argument('-out','--outDir',help='Short version', default = 'dtiPreproc')
    args = parser.parse_args()

    # update apDir and paDir
    args.apDir = join(args.dir, 'DTI_AP')
    args.paDir = join(args.dir, 'DTI_PA')
    #args.outDir = join(args.dir, 'dtiPreproc')

    subDirs = os.listdir(args.directory)
    dtiDirs = [x for x in subDirs \
            if re.search('DTI_', x, re.IGNORECASE)]
    dtiDirsCount = len(dtiDirs)

    # If there are directories named 'DTI_AP' and 'DTI_PA'
    if dtiDirsCount > 1 and isdir(args.apDir) and isdir(args.paDir):
        try:
            ap_nifti, ap_bvec, ap_bval = get_dti_trio(args.apDir)
        except:
            sys.exit('AP nifti file is missing')

        try:
            pa_nifti = get_nifti(os.path.join(args.directory, i))
        except:
            sys.exit('PA nifti file is missing')

        # dtiPreproc(APdata, bvals, bvecs, PAdata)
        dtiPreproc(ap_nifti,
                   ap_bvec,
                   ap_bval,
                   pa_nifti,
                   args.outDir)

    # If only one encoding direction was used
    elif dtiDirsCount < 2:
        if not args.old:
            print 'There are less than two DTI directories'
            print 'Please give a root location with two DTI subdirectories (AP & PA)'
            print ''.join(dtiDirs)
            sys.exit()
        else:
            dtiPreproc_old(args)
    else:
        print 'There are more than two DTI directories'
        print 'Please give a root location with only two DTI subdirectories (AP & PA)'
        for i in dtiDirs:
            print '>', i
        sys.exit()

