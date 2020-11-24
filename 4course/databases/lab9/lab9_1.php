<?php
	$link = mysqli_connect("localhost", "root", "root", "sample");
	if (mysqli_connect_errno()) {
		printf("Не удалось подключиться: %s\n", mysqli_connect_error());
    	exit();
	}
	mysqli_query($link, "DROP TABLE IF EXISTS notebook_br06");
	$sql = "CREATE TABLE notebook_br06 (id int not null auto_increment primary key, name varchar(50), city varchar(50), address varchar(50), birthday date, mail varchar(20))";
	if (mysqli_query($link, $sql)) {
	  print "Таблица успешно создана";
	} else {
	  print "Ошибка при создании таблицы: " . mysqli_error($link);
	}
	mysqli_close($link);
?>