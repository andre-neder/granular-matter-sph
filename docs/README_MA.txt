LaTeX-Dokumentenklasse fuer Bachelorarbeiten der AG Computergraphik (v1.9)
=====================================================================

Die Dokumentenklassen "cgMA.cls" und "cgMAgruppe.cls" basieren auf der Standard-Klasse "article" und dienen zur Vereinheitlichung des Erscheinungsbildes der Masterarbeiten der Arbeitsgruppe Computergraphik im Rahmen des Styleguides des Fachbereichs durch den Pruefungsausschuss. Dazu stellen sie ein Deckblatt- Template zur Verfuegung und bindet einige wichtige LaTeX-Pakete (babel, pslatex, graphicx) ein.

Der Editor sollte auf die Verwendung von UTF-8 eingestellt sein! Die Kodierung kann für bestehende Projekte in der *.cls Datei geändert werden: \RequirePackage[utf8]{inputenc}  ->  \RequirePackage[latin1]{inputenc}

Um die Dokumentenklasse einzubinden, verwendet man den Befehl

     \documentclass[<optionen>]{cgMA}

Dabei werden folgende Optionen unterstuetzt (Reihenfolge beliebig):

twoside		-  Fuer doppelseitigen Ausdruck   
intern|extern	-  Layout fuer interne oder externe Arbeiten
times|palatino	-  Schriftart TimesNewRoman oder Palatino

Die Schriftartoptionen sind dabei exklusiv, bei mehreren wird nur die zuletzt angegebene verwendet. Die Optionen koennen auch ganz weggelassen werden, das Dokument wird dann fuer den einseitigen Druck als interne Arbeit gesetzt und als Schriftart die TeX-Standardschrift (Computer Modern) verwendet.

Zum Parametrisieren der Deckblatt-Vorlage koennen die folgenden Befehle im Vorspann (d.h. vor dem \begin{document}) verwendet werden:

     \title{Titel der Arbeit}
     \author{Name des Autors}
     
     \erstgutachter{Name des Betreuers (inkl. akadem. Grad)}
     \erstgutachterInfo{Institut/Arbeitsgruppe, ggf. externe Institution}
     
     \zweitgutachter{Name des Betreuers (inkl. akadem. Grad)}
     \zweitgutachterInfo{Institut/Arbeitsgruppe, ggf. externe Institution}
     
     \externLogo{breite}{Logo, Dateiname mit relativem Pfad ohne Endung}
     \externName{Name der externen Institution}

Dabei sind \title{...} und \author{...} zwingend erforderlich. Die Angabe des Erstgutachters kann dabei entfallen, da Prof. Mueller in der Regel auch der Erstgutachter der Arbeit ist; der Zweitgutachter muss gemaess der Pruefungsordnung jedoch angegeben werden. 
Zusaetzlich koennen bei externen Institutionen deren Logo und Name der Abteilung, Arbeitsgruppe usw. mit den Befehlen \externLogo und \externName angegeben werden. Das Logo sollte in den Formaten PDF (fuer PDF-Dokumente) und EPS (fuer Postscript) vorliegen; bei Verwendung von pdflatex zur ausschliesslichen Erzeugung von PDF-
Dokumenten koennen zusaetzlich auch JPG- und PNG-Grafiken eingebunden werden. 


Die Titelseite mit den gemachten Angaben kann dann innerhalb des Dokumentes durch den Befehl

     \maketitle

erzeugt werden, wobei bei zweiseitigem Ausdruck automatisch eine
zusaetzliche Leerseite (Rueckseite des Deckblattes) generiert wird. Ausserdem wird eine Erklaerung zu den Quellen und der Veroeffentlichung der Arbeit auf der folgenden Seite eingefuegt.

Für Arbeiten in englischer Sprache muss vor Beginn des eigenen Dokumentes die Sprache umgeschaltet werden. Dazu wird mit Hilfe des Kommandos aus dem babel-Paket

	\selectlanguage{english}
	
die Sprache ausgewählt. Es werden automatisch die Sprachen Englisch (english) Deutsch (ngerman) aktiviert.



Zum Abschluss ein Beispiel fuer die Verwendung der Dokumentenklasse:

\documentclass[twoside,extern,palatino]{cgMA}

\author{Martina Mustermann}
\title{Automatisches Generieren von Musterbildern mit Hilfe der Grafikhardware}
\zweitgutachter{Dr. K. Nickel}
\zweitgutachterInfo{Deutsches Institut f{\"u}er Normung, Abteilung Moderne Technologien}
\externLogo{7.46cm}{logos/UniLogoNeu}
\externName{DIN: NewTechnologies}

\begin{document}

% Umschalten der Sprache (für englische Rubrikbezeichnungen etc.)
%\selectlanguage{english}


\maketitle

\newpage

\pagenumbering{roman}
\tableofcontents
\clearpage         % oder \cleardoublepage bei zweiseitigem Druck
% \listoffigures   % fuer ein eventuelles Abbildungsverzeichnis
% \clearpage
\pagenumbering{arabic}


% Hier kommt jetzt der eigentliche Text der Arbeit


\end{document}