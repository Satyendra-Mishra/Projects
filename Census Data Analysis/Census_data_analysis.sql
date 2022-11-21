--display all user databases
SELECT name FROM master.dbo.sysdatabases
WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb');

-- create database
CREATE DATABASE Census;
USE Census;

-- Display all the tables in the current database
SELECT TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG='Census';


--############## Exploring the Table - AreaPop ####################

-- Area & Population table
SELECT * FROM dbo.AreaPop;

-- columns in the table AreaPop
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = N'AreaPop';

-- Duplicate records
SELECT COUNT(*) AS Row_count FROM dbo.AreaPop;
SELECT DISTINCT COUNT(*) FROM dbo.AreaPop;

SELECT State, District from dbo.AreaPop
GROUP BY State, District
HAVING COUNT(*) > 1;

select * from dbo.AreaPop
where state = 'Maharashtra' AND District = 'Aurangabad';

-- Count of null values in the table AreaPop
SELECT
     SUM(CASE WHEN District IS NULL THEN 1 ELSE 0 END) AS District,
	 SUM(CASE WHEN State IS NULL THEN 1 ELSE 0 END) AS State,
	 SUM(CASE WHEN Area_km2 IS NULL THEN 1 ELSE 0 END) AS Area_km2,
	 SUM(CASE WHEN Population IS NULL THEN 1 ELSE 0 END) AS Population
FROM dbo.AreaPop;

-- 33 missing values in State column

-------------------------------------------------------------------

--############## Exploring the Table - GrateLitRate ####################

-- GrateLitRate table
SELECT * FROM dbo.GrateLitRate;

-- Columns in the table 
SELECT COLUMN_NAME 
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = N'GrateLitRate';

-- erroneous records
SELECT State, District FROM dbo.GrateLitRate
GROUP BY State, District
HAVING COUNT(*) > 1;

-- Count of missing values 
SELECT
     SUM(CASE WHEN District IS NULL THEN 1 ELSE 0 END) AS District,
	 SUM(CASE WHEN State IS NULL THEN 1 ELSE 0 END) AS State,
	 SUM(CASE WHEN Growth IS NULL THEN 1 ELSE 0 END) AS Growth,
	 SUM(CASE WHEN Sex_Ratio IS NULL THEN 1 ELSE 0 END) AS Sex_Ratio,
	 SUM(CASE WHEN Literacy IS NULL THEN 1 ELSE 0 END) AS Literacy
FROM dbo.GrateLitRate;

-- No null values in the GrateLitRate table
 

--######################### Analysis ######################

--Q1. Find the Average growth rate, Sex ratio & Avg literacy rate state-wise.
SELECT State,
       ROUND(AVG(Growth),2) as average_growth_rate,
       AVG(Sex_Ratio) as average_sex_Ratio,
       ROUND(AVG(Literacy),2) as average_literacy_rate
FROM dbo.GrateLitRate
GROUP BY State;


--Q2. Find the Top 3 and Bottom 3 states for literacy in the same query.

SELECT state, avg_lit, 
       CASE 
           WHEN t.TopThree <= 3 THEN 'Top Three'
	       WHEN t.BottomThree <= 3 THEN 'Bottom Three'
       END AS Position
FROM 
(SELECT  state, AVG(literacy) avg_lit, 
         ROW_NUMBER() OVER (ORDER BY AVG(literacy)) AS BottomThree,
         ROW_NUMBER() OVER (ORDER BY AVG(literacy) DESC) AS TopThree
 FROM GrateLitRate
 GROUP BY State
) AS t
WHERE t.TopThree <=3 or t.BottomThree <=3


-- Q3. Total males and females

SELECT SUM(Females) AS Total_Females, SUM(Males) AS Total_Males FROM
( SELECT t1.state, t1.district, t1.Sex_Ratio, t2.Population,
         ROUND((t1.Sex_Ratio/(1000.0+t1.Sex_Ratio))*t2.Population,0) AS Females, 
         ROUND((1000.0/(1000.0+t1.Sex_Ratio))*t2.Population,0) AS Males
  FROM GrateLitRate t1
  LEFT JOIN AreaPop t2
  ON t1.district = t2.district
) AS t;

-- Q4. Population in the previous census.

SELECT SUM(Last_year_census) AS Total_Previous_Population FROM
(  SELECT t1.state, t1.district, t1.Growth, t2.Population,
          ROUND((t2.Population/(1.0+t1.Growth)),0) AS Last_year_census
   FROM GrateLitRate t1
   LEFT JOIN AreaPop t2
   ON t1.district = t2.district
) AS t;


-- Q5 Find the Top 3 districts are having high literacy rates from each state.

SELECT state, district, Literacy, rno from
(
   SELECT State, District, Literacy, 
          ROW_NUMBER() OVER (PARTITION BY State ORDER BY Literacy DESC) as rno
   FROM GrateLitRate
)AS q
WHERE q.rno <=3