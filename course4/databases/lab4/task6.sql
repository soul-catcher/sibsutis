# accept v_sname char prompt 'Enter name: ';
select sname, city, comm, count(onum), sum(amt) from sal inner join ord o using(snum)
#where sname = '&v_sname'
where sname = 'Peel'
group by snum;
