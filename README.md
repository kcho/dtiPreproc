#DTI preprocessing script

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

- dicomConversion(outputDir,DTIdirectories)
 
- nameChange(outputDir)
 
- extractB0images(outputDir)
 
- writeAcqParams(outputDir)
 
- makeEvenNumB0(outputDir)

################################################
# Running topup
################################################
topup(outputDir)
