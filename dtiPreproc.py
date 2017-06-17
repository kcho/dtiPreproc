#!/ccnc_bin/venv/bin/python
from __future__ import division
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
    print('\tMake the slice number even : ', basename(niftiImg))

    # Load nifti file
    f = nb.load(niftiImg)
    data = f.get_data()
    sliceNum = data.shape[2]

    # Make the slice number even
    if not sliceNum % 2 == 0:
        print('\t\tRemoving the most inferior slice')
        # 3D
        try:
            newData = data[:,:,1:]
        # 4D
        except:
            newData = data[:,:,1:,:]

        # save data
        even_niftiImg = join(outdir, 'even_{0}'.format(basename(niftiImg)))
        img = nb.Nifti1Image(newData, f.affine)
        img.to_filename(even_niftiImg)
        return even_niftiImg

    else:
        return niftiImg

def getmat_b0(niftiImg, bval):
    '''
    Return B0 images in 4d nibabel format,
    extracted from the data.nii.gz using the index in the bval
    '''

    # Load images
    f = nb.load(niftiImg)
    data = f.get_data()

    # Load bval if there is a location given
    if bval:
        with open(bval, 'r') as bvalf:
            bvals = bvalf.read().split(' ')

        bvalsArray = np.array(bvals)
        b0_indexArray = np.where(bvalsArray=='0')[0]

    b0_data = data[:,:,:,b0_indexArray]
    img = nb.Nifti1Image(b0_data, f.affine)
    return img

#def extractMeanB0images(niftiImg,bval, outDir):
    #img = nb.load(niftiImg)
    #imgData = img.get_data()

    #if bval:
        #with open(bval, 'r') as f:
            #bvals = f.read().split(' ')

        #bvalsArray = np.array(bvals)
        #b0_indexArray = np.arange(len(bvalsArray))[bvalsArray=='0']

    #else:
        #try:
            #b0_indexArray = range(imgData.shape[3])
        #except:
            #b0_indexArray = [0]

    #all_b0_imgData = np.zeros_like(imgData[:,:,:,0])
    #for b0_index in b0_indexArray:
        #b0_data = imgData[:,:,:,b0_index]
        #all_b0_imgData = all_b0_imgData + b0_data

    #mean_b0_image = all_b0_imgData / len(b0_indexArray)
    #newData = nb.Nifti1Image(mean_b0_image, img.affine)

    #imgName = os.path.basename(niftiImg)
    #outImgLoc = os.path.join(outDir, imgName)
    #newData.to_filename(outImgLoc)

    #return outImgLoc
    

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


def writeAcqParams(ap_b0_num, pa_b0_num, matrix_size, echo_spacing, outDir,full):
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
    num = (matrix_size-1) * (1/1000 * echo_spacing)
    acqparamLoc = join(outDir, 'acqparams.txt')

    #if not isfile(acqparamLoc):
    if isfile(acqparamLoc):
        # Writing acqparams.txt
        ap_array = np.tile([0, -1, 0, num], ap_b0_num).reshape(ap_b0_num, 4)
        pa_array = np.tile([0, 1, 0, num], pa_b0_num).reshape(pa_b0_num, 4)
        concat_array = np.concatenate([ap_array, pa_array])
        np.savetxt(acqparamLoc, concat_array, 
                   fmt=['%d', '%d', '%d', '%0.3f'])
    
    return acqparamLoc


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



def dtiPreproc(ap_nifti, ap_bvec, ap_bval, pa_nifti, pa_bvec, pa_bval, outDir):
    # make the number of z slice even
    ap_nifti_even = makeEvenNumB0(ap_nifti, outDir)
    pa_nifti_even = makeEvenNumB0(pa_nifti, outDir)

    # get b0 maps in 4d nibabel format
    ap_b0_nifti = getmat_b0(ap_nifti, ap_bval)
    pa_b0_nifti = getmat_b0(pa_nifti, pa_bval)

    ap_b0_data = ap_b0_nifti.get_data()
    pa_b0_data = pa_b0_nifti.get_data()

    # Merge ap & pa b0 images
    # Save to nifti
    merged_b0 = np.concatenate([ap_b0_data, pa_b0_data], axis=3)
    merged_b0_loc = join(outDir,'merged_b0.nii.gz')
    #nb.Nifti1Image(merged_b0, ap_b0_nifti.affine).to_filename(merged_b0_loc)
    
    # get matrix
    matrix_size = ap_b0_data.shape[0] 

    # get echo spacing
    echo_spacing = 0.77 # KJS2 protocol

    acqparam = writeAcqParams(ap_b0_data.shape[3], 
                              pa_b0_data.shape[3],
                              matrix_size,
                              echo_spacing,
                              outDir,False)

    #topup(merged_b0_all, acqparam, outDir)
    #applytopup(ap_b0_list[0], pa_b0_list[0], acqparam, outDir)
    #eddy_out, mask = eddy(ap_nifti_even, ap_bval, ap_bvec, acqparam, outDir)
    #dtifit(eddy_out, mask, ap_bvec, ap_bval, 'dti', outDir)


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

