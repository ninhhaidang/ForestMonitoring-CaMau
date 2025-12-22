%% vnuvi.bst - BibTeX style for VNU thesis (Vietnamese references)
%% Uses "tr." for pages instead of "pp."
%% Sorts by author name as-is (for Vietnamese names sorted by given name)

ENTRY
  { address
    author
    booktitle
    chapter
    edition
    editor
    howpublished
    institution
    journal
    key
    month
    note
    number
    organization
    pages
    publisher
    school
    series
    sortname
    title
    type
    url
    volume
    year
  }
  {}
  { label }

INTEGERS { output.state before.all mid.sentence after.sentence after.block }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}

STRINGS { s t }

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
        { add.period$ write$
          newline$
          "\newblock " write$
        }
        { output.state before.all =
            'write$
            { add.period$ " " * write$ }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {fin.entry}
{ add.period$
  write$
  newline$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}

FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "\textit{" swap$ * "}" * }
  if$
}

FUNCTION {enquote}
{ duplicate$ empty$
    { pop$ "" }
    { "``" swap$ * "''" * }
  if$
}

INTEGERS { nameptr namesleft numnames }

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr "{ff}{vv}{ll}{jj}" format.name$ 't :=
      nameptr #1 >
        { namesleft #1 >
            { ", " * t * }
            { t "others" =
                { " và cộng sự" * }
                { ", " * t * }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {format.authors}
{ author empty$
    { "" }
    { author format.names }
  if$
}

FUNCTION {format.editors}
{ editor empty$
    { "" }
    { editor format.names
      editor num.names$ #1 >
        { " (Chủ biên)" * }
        { " (Chủ biên)" * }
      if$
    }
  if$
}

FUNCTION {format.title}
{ title empty$
    { "" }
    { title enquote }
  if$
}

FUNCTION {format.btitle}
{ title empty$
    { "" }
    { title emphasize }
  if$
}

FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" = not
            { "--" *
              t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
                { "-" *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
          if$
        }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}

FUNCTION {format.date}
{ year empty$
    { "" }
    { year }
  if$
}

FUNCTION {format.journal.vol.num.year.pages}
{ journal empty$
    { "" }
    { journal emphasize
      volume empty$
        'skip$
        { ", tập " * volume * }
      if$
      number empty$
        'skip$
        { ", số " * number * }
      if$
      year empty$
        'skip$
        { ", " * year * }
      if$
      pages empty$
        'skip$
        { ", tr. " * pages n.dashify * }
      if$
    }
  if$
}

FUNCTION {format.booktitle.pages}
{ booktitle empty$
    { "" }
    { booktitle emphasize
      pages empty$
        'skip$
        { ", tr. " * pages n.dashify * }
      if$
    }
  if$
}

FUNCTION {format.publisher.address.year}
{ publisher empty$
    { "" }
    { publisher
      address empty$
        'skip$
        { ", " * address * }
      if$
      year empty$
        'skip$
        { ", " * year * }
      if$
    }
  if$
}

FUNCTION {format.address.year}
{ address empty$
    { year empty$
        { "" }
        { year }
      if$
    }
    { address
      year empty$
        'skip$
        { ", " * year * }
      if$
    }
  if$
}

FUNCTION {format.url}
{ url empty$
    { "" }
    { "Truy cập: \url{" url * "}" * }
  if$
}

FUNCTION {article}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  format.journal.vol.num.year.pages "journal" output.check
  note output
  fin.entry
}

FUNCTION {book}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check }
    { format.authors output.nonnull }
  if$
  format.btitle "title" output.check
  edition empty$
    'skip$
    { "Tái bản lần " edition * output }
  if$
  format.publisher.address.year "publisher" output.check
  note output
  fin.entry
}

FUNCTION {inbook}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check }
    { format.authors output.nonnull }
  if$
  format.btitle "title" output.check
  format.publisher.address.year "publisher" output.check
  pages empty$
    'skip$
    { "tr. " pages n.dashify * output }
  if$
  note output
  fin.entry
}

FUNCTION {incollection}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  "Trong: " booktitle emphasize * output
  format.publisher.address.year output
  pages empty$
    'skip$
    { "tr. " pages n.dashify * output }
  if$
  note output
  fin.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  format.booktitle.pages "booktitle" output.check
  publisher empty$
    'skip$
    { publisher output }
  if$
  year output
  note output
  fin.entry
}

FUNCTION {conference} { inproceedings }

FUNCTION {manual}
{ output.bibitem
  author empty$
    { organization empty$
        'skip$
        { organization output.nonnull }
      if$
    }
    { format.authors output.nonnull }
  if$
  format.btitle "title" output.check
  organization output
  format.address.year output
  note output
  fin.entry
}

FUNCTION {mastersthesis}
{ output.bibitem
  format.authors "author" output.check
  format.btitle "title" output.check
  "Luận văn thạc sĩ" output.nonnull
  school "school" output.check
  format.address.year output
  note output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  format.authors output
  format.btitle output
  publisher output
  format.address.year output
  format.url output
  note output
  fin.entry
}

FUNCTION {phdthesis}
{ output.bibitem
  format.authors "author" output.check
  format.btitle "title" output.check
  "Luận án tiến sĩ" output.nonnull
  school "school" output.check
  format.address.year output
  note output
  fin.entry
}

FUNCTION {proceedings}
{ output.bibitem
  editor empty$
    { organization output }
    { format.editors output.nonnull }
  if$
  format.btitle "title" output.check
  format.publisher.address.year output
  note output
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  format.authors "author" output.check
  format.btitle "title" output.check
  institution "institution" output.check
  type empty$
    { "Báo cáo kỹ thuật" output.nonnull }
    { type output.nonnull }
  if$
  number output
  format.address.year output
  note output
  fin.entry
}

FUNCTION {unpublished}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  note "note" output.check
  fin.entry
}

FUNCTION {default.type} { misc }

READ

FUNCTION {sortify}
{ purify$
  "l" change.case$
}

FUNCTION {presort}
{ sortname empty$
    { author empty$
        { key empty$
            { "to sort, need sortname, author or key in " cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { author sortify }
      if$
    }
    { sortname sortify }
  if$
  "    "
  *
  year field.or.null sortify
  *
  "    "
  *
  title field.or.null
  sortify
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}

ITERATE {presort}

SORT

STRINGS { longest.label }

INTEGERS { number.label longest.label.width }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #1 'number.label :=
  #0 'longest.label.width :=
}

FUNCTION {longest.label.pass}
{ number.label int.to.str$ 'label :=
  number.label #1 + 'number.label :=
  label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
}

EXECUTE {initialize.longest.label}

ITERATE {longest.label.pass}

FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\noindent{\fontsize{14pt}{16pt}\selectfont\textbf{Tiếng Việt}}" write$ newline$
  "\vspace{0.5em}" write$ newline$
  "\begin{thebibliography}{"  longest.label  * "}" * write$ newline$
}

EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}
