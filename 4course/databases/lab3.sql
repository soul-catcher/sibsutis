#1
select onum, amt * 0.8 amount, '€' currency, sname, comm
from ord
         inner join sal using (snum)
where odate like '03-OCT%';

#2
select onum, sname, cname, sal.city, cust.city
from ord
         left outer join sal using (snum)
         left outer join cust using (cnum)
where sal.city in ('London', 'Rome')
   or cust.city in ('London', 'Rome')
order by onum;

#3
select sname, sum(amt), comm
from sal
         inner join ord using (snum)
where odate < '05-OCT'
group by snum;

#4
select onum, amt, sname, cname, sal.city, cust.city
from ord
         inner join sal using (snum)
         inner join cust using (cnum)
where sal.city between 'L%' and 'R%'
  and cust.city between 'L%' and 'R%';

#5
select a.cname, b.cname
from cust a,
     cust b
where a.snum = b.snum
  and a.cnum < b.cnum;

#6
select cname
from cust
where snum in (
    select snum
    from sal
    where comm < .13
);

#7
select sum(ord.amt), sal.sname
from ord,
     sal
where ord.snum = sal.snum
group by sal.snum, sal.sname
having sum(ord.amt) > max(ord.amt);
