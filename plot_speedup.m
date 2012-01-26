function [h, p, g, c, s] = plot_speedup(log_file, ckt)
	for r = 1:length(log_file); 
		p(r) = log_file(r,1) / 1024;
		g(r) = log_file(r,3)+log_file(r,2);
		c(r) = log_file(r,4);
		s(r) = log_file(r,4)/g(r);

	endfor
	h = plot(p, s, "x-");
	titlename = ["Speedup of ", ckt];
	title(titlename);
	xlabel("Number of Test Vectors (2^{10})");
	ylabel("Speedup Factor");
endfunction
