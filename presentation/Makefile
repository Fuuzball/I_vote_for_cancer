LATEX=pdflatex -halt-on-error -interaction=nonstopmode
BIBTEX=bibtex

all: slides.pdf

verbose:
	${LATEX} slides.tex

slides.pdf: slides.tex
	${LATEX} $< > /dev/null
#	${BIBTEX} $(basename $<) > /dev/null
	${LATEX} $< > /dev/null
#	${LATEX} $< > /dev/null
#	${LATEX} $< > /dev/null
	rm -f $(basename $<).aux $(basename $<).log $(basename $<).nav \
		$(basename $<).out $(basename $<).snm $(basename $<).toc
