insert into sal_copy
values (1008, 'Barmen', '100-rentgen', 0.7),
       (1009, 'Saharov', 'Yantar', 0.9);
select * from sal_copy;
# spool results.txt;
# spool off;
update sal_copy set comm = comm * 2;
select * from sal_copy;
