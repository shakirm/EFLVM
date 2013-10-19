function printPDF(outName, outDir, print)

outName = [outName,'.eps'];
if print
    exportfig(gcf,outName,'bounds','tight','color','rgb','lockAxes',0);
    eps2pdf(outName)
    system(sprintf('mv %s %s/pdf',outName,outDir));
    ix = strfind(outName,'.');
    system(sprintf('mv %s %s/pdf',strcat(outName(1:ix),'pdf'),outDir));
end;