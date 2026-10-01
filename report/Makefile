TEX = informe_siniestros_viales_envigado.tex
PDF = $(TEX:.tex=.pdf)

all: $(PDF)

$(PDF): $(TEX)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

clean:
	latexmk -C $(TEX)

.PHONY: all clean