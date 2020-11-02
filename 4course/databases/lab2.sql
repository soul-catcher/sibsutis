select * from cust
where left(cname, 1) between 'A' and 'G';

select * from sal
where sname like '%e%';

select sum(amt) from ord
where odate = '03-OCT-90';

select count(distinct city) from cust;

select cname, o.onum, min(o.amt), o.odate, o.cnum, o.snum
from cust c inner join ord o using (cnum)
group by cnum;

select cnum, max(cname) cname, city, rating, snum from cust
where left(cname, 1) = 'G';

select city, max(rating) rating from cust
group by city;
