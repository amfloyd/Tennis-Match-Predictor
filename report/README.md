# How to Compile

To compile this report, do the following:

* Run `bibtex report`. This will generate the bibliography
* Run `pdflatex report.tex` twice. This will generate the report without the bibliography on the first pass, and then insert the bibliography and resolve all citations on the second pass. 
* If you only want to generate the text in the report, it is sufficient to run `pdflatex report.tex`
