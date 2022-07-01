-- Creation of papers table
DROP TABLE IF EXISTS public."papers" CASCADE;
CREATE TABLE public."papers"
(
  name text,
  scholar_link text,
  doi text,
  bibtext bool,
  pdf_name text,
  year int,
  scholar_page text,
  journal text,
  downloaded bool,
  downloaded_from text,
  authors text
);

\copy public."papers" FROM '/var/lib/pg_data/result.csv' DELIMITER ',' CSV HEADER;