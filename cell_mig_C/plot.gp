reset

#set term gif animate
#set output 'foobar.gif'

# define fixed axis-ranges
#set xrange [-25:25]
#set yrange [-25:25]
#set zrange [-1:1]

# filename and n=number of lines of your data 
#filedata = 'data.dat'
#n = system(sprintf('cat %s | wc -l', filedata))

#do for [j=1:n] {
#    set title 'time '.j
#    splot filedata u 3:1:2 every ::1::j w p
    #splot filedata u 2:3:4 every ::1::j w l lw 2, \
    #      filedata u 2:3:4 every ::j::j w p pt 7 ps 2
#}

set terminal gif animate 

set output 'foobar.gif'
datafile = "data.dat"

speed = 1

set xrange [-100:100]
set yrange [-100:100]
#set zrange [0:1]

do for [i=1:10000]{
plot datafile index speed*(i-1) u 1:2:($3 == 0 ? NaN : $3) with image notitle
#plot datafile nonuniform matrix index speed*(i-1) u 1:2:3 with image notitle
#plot datafile nonuniform index speed*(i-1) u 1:2:3 with image notitle
#splot datafile index speed*(i-1) u 1:2:3 notitle
#splot datafile index speed*(i-1) u 1:2:($3 == 0 ? NaN : $3) with points lt 6 notitle
#splot datafile index speed*(i-1) u 1:2:3 with points lt 6 notitle
#plot datafile index speed*(i-1) u 1:3 with points lt 6 notitle
}



#

    #splot datafile index 5*(i-1) u 1:2:3 with dots lc rgb "black" notitle
    #plot datafile index 5*(i-1) u 1:3 with points lc rgb "black" notitle

