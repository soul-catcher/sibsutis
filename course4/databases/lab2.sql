select * from cust
where cname between 'A%' and 'G%';

select * from sal
where sname like '%e%';

select sum(amt) from ord;

select count(distinct city) from cust;

select cname, o.*
from cust c inner join ord o on c.cnum = o.cnum
where amt = (
    select min(amt)
    from ord
    where cnum = c.cnum
);
