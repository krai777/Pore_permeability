#!/bin/bash
# Create a counter starting from -30
counter=-30
# Loop through trajectory files 0 to 61
for i in {0..61}; do
    # Create TCL script for each trajectory
    cat > analysis1_${i}.tcl << EOF
#####Script developed by Kunal Rai#########
##for more detail read README####
#upload system psf 
mol new ../../input/lipid_aqp_wi.psf
#upload trajectory file
mol addfile ../windows_traj/traj_${i}.dcd waitfor all

# Define the atom selections
set sel_glycerol [atomselect top "residue 34649"]
set sel_membrane [atomselect top "resname POPC"]

# Open the output file for writing
set outfile [open "z1_${counter}.dat" w]

# Write the header to the file
puts \$outfile "time z_relative"

# Loop over all frames in the trajectory
set num_frames [molinfo top get numframes]
for {set frame 0} {\$frame < \$num_frames} {incr frame} {
    # Set the current frame
    animate goto \$frame
    
    # Update the atom selections
    \$sel_glycerol frame \$frame
    \$sel_membrane frame \$frame
    
    # Calculate time (in nanoseconds)
    set time [expr \$frame * 0.214]
    
    # Get the z-coordinate of center of mass of glycerol
    set z_glycerol [lindex [measure center \$sel_glycerol weight mass] 2]
    
    # Get the z-coordinate of center of mass of membrane (POPC)
    set z_membrane [lindex [measure center \$sel_membrane weight mass] 2]
    
    # Calculate relative z-position (glycerol - membrane center)
    set z_relative [expr \$z_glycerol - \$z_membrane]
    
    # Write the time and relative z-coordinate to the file
    puts \$outfile "\$time \$z_relative"
}

# Close the output file
close \$outfile

# Delete the atom selections
\$sel_glycerol delete
\$sel_membrane delete

puts "Data has been written to z_${counter}.dat"
quit
EOF
    # Run VMD with the TCL script
    vmd -dispdev text -e analysis1_${i}.tcl
    
    # Increment counter
    counter=$((counter + 1))
done
