-- table headers
-- index snp	snp.chr	snp.pos	A1	A2	eaf.eu	eaf.sa	snp.gene	cpg	cpg.chr	cpg.pos	cpg.gene	discovery.population	beta.eur	se.eur	p.eur	beta.sa	se.sa	p.sa	beta.combined	se.combined	p.combined

-- select count(*) from cosmopolitan_meqtl;
-- SELECT `index`, `snp`, `snp.pos`, `cpg`, `cpg.pos`, ABS(`snp.pos` - `cpg.pos`) as `delta` FROM cosmopolitan_meqtl WHERE  `delta` <= 1000;
-- WITH temp AS (
-- 	SELECT `index`, `SNP`, `SNP.Pos`, `CpG`, `CpG.Pos`, ABS(`SNP.Pos` - `CpG.Pos`) as `DELTA` FROM cosmopolitan_meqtl WHERE  `DELTA` <= 1000
-- )	SELECT * FROM temp;

WITH temp AS (
	SELECT `index`, `SNP`, `SNP.Pos`, `CpG`, `CpG.Pos`, ABS(`SNP.Pos` - `CpG.Pos`) as `DELTA` FROM cosmopolitan_meqtl WHERE  `DELTA` <= 10000
)
	SELECT *
	FROM temp
    GROUP BY `SNP`
	ORDER BY `index` ASC
	;
