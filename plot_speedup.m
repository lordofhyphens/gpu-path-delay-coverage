function plot_speedup(log_file, ckt)
	for r = 1:length(log_file); 
		c(r) = log_file(r,1);
		z(r) = log_file(r,3)+log_file(r,2);
		s(r) = log_file(r,4)/z(r);

	endfor
	plot(c, s);
	titlename = ["Speedup of ", ckt];
	title(titlename);
	xlabel("Number of Test Vectors");
	ylabel("Speedup Factor");
endfunction
