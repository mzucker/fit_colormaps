import glob
import os
import sys

import numpy as np
from PIL import Image


import skimage.measure
import skimage.filters

BLUR_SIGMA = 1.5

NUM_POINTS_OUT = 256

PDF_OUTPUT = True

if PDF_OUTPUT:
    import cairo

######################################################################

def make_pdf(named_shapes):

    INCH = 72

    ncols = 3
    nrows = 4
    
    page_dims = np.array((8.5, 11))*INCH
    cell_counts = np.array((3, 4))
    
    margin = 0.5*INCH
    spacing = 0.25*INCH

    cell_dims = (page_dims - 2*margin - (cell_counts-1)*spacing)/cell_counts
    cw, ch = cell_dims

    text_height = 16
    font_size = 14
    text_yoffset = 2

    surface = cairo.PDFSurface('shapes.pdf', *page_dims)
    ctx = cairo.Context(surface)
    ctx.set_font_size(font_size)

    cur_row = 0
    cur_col = 0

    base_offset = (0.5 * page_dims -
                   0.5*(cell_counts*cell_dims + (cell_counts-1)*spacing))
    
    for name, shape in named_shapes:

        cur_pos = (cur_col, cur_row)
        cell_offset = 0.5*cell_dims + (cell_dims + spacing)*cur_pos

        ctx.save()
        ctx.translate(*(base_offset + cell_offset))

        prng = shape.max(axis=0) - shape.min(axis=0)
        pscl = ((cell_dims - (0, text_height)) / prng).min()
        
        
        extents = ctx.text_extents(name)
        ctx.move_to(-0.5*extents.width, 0.5*ch - text_yoffset)
        ctx.show_text(name)
        ctx.fill()

        ctx.translate(0, -0.5*text_height)

        ctx.scale(pscl, -pscl)
        ctx.translate(-0.5, -0.5)
        
        ctx.move_to(*shape[0])
        for p in shape[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.restore()
        ctx.fill()

        cur_col += 1
        if cur_col >= ncols:
            cur_col = 0
            cur_row += 1
            if cur_row >= nrows:
                ctx.show_page()
                cur_row = 0

######################################################################

def get_bbox_area(contour):
    p0 = contour.min(axis=0)
    p1 = contour.max(axis=0)
    return (p1-p0).prod()

######################################################################

def sample_contour_uniformly(contour):

    # get length of each edge along contour
    nctr = len(contour)
    idx = np.arange(nctr+1) % nctr
    deltas = contour[idx[1:]] - contour[idx[:-1]]
    edge_lengths = np.linalg.norm(deltas, axis=1)

    # get the perimeter
    perimeter = edge_lengths.sum()

    # get edge output index (will be output index for each edge
    # from 0 to NUM_POINTS_OUT)
    edge_indices = np.cumsum(edge_lengths)
    edge_indices *= NUM_POINTS_OUT / perimeter
    edge_indices = np.hstack(([0], edge_indices))

    # force last index to be NUM_POINTS_OUT
    assert np.isclose(edge_indices[-1], NUM_POINTS_OUT)
    edge_indices[-1] = NUM_POINTS_OUT

    # shape always starts at first point
    shape = [ contour[0] ]

    # for remaining points
    for i in range(1, NUM_POINTS_OUT):

        # find first item in edge_indices greater than i
        eidx = (edge_indices > i).argmax()

        # sanity checks
        assert eidx > 0
        assert edge_indices[eidx] > i
        assert edge_indices[eidx-1] <= i

        # distance along this edge
        e0 = edge_indices[eidx-1]
        e1 = edge_indices[eidx]
        u = (i - e0) / (e1 - e0)
        assert u >= 0 and u <= 1

        # linearly interpolate along edge to get point
        c0 = contour[eidx-1]
        c1 = contour[eidx%nctr]
        p = c0 + u*(c1 - c0)

        # append point to shape
        shape.append(p)

    # convert to array
    shape = np.array(shape)

    return shape

######################################################################

def line_segment_dists(a, b, p):

    ba = b-a
    pa = p-a

    bapa = (ba*pa).sum(axis=1)
    baba = (ba*ba).sum(axis=1)

    u = np.clip(bapa / baba, 0.0, 1.0)

    diff = pa - u.reshape(-1, 1)*ba

    return np.linalg.norm(diff, axis=1)


######################################################################

def sample_contour_greedily(contour):

    n = len(contour)
    idx = np.arange(n)

    prev_idx = (idx + n - 1) % n
    next_idx = (idx + 1) % n

    seg_dists = line_segment_dists(contour[prev_idx],
                                   contour[next_idx],
                                   contour)
    
    remove_immediately = (seg_dists == 0.0)

    if np.any(remove_immediately):
        print('removing {} of {} points immediately!'.format(remove_immediately.sum(), n))
        keep = ~remove_immediately
        contour = contour[keep]
        n = len(contour)
        idx = np.arange(n)
        prev_idx = (idx + n - 1) % n
        next_idx = (idx + 1) % n
        seg_dists = line_segment_dists(contour[prev_idx],
                                       contour[next_idx],
                                       contour)
        
    
    active_idx = list(range(n))

    while len(active_idx) > NUM_POINTS_OUT:

        # get index of thing to remove within active set
        idx_within_active = seg_dists[active_idx].argmin()

        # get index of thing to remove within original arrays
        idx_within_orig = active_idx[idx_within_active]

        # remove from active set
        active_idx.pop(idx_within_active)

        # update linked list pointers for neighbors of removed
        prev_of_removed = prev_idx[idx_within_orig]
        next_of_removed = next_idx[idx_within_orig]

        next_idx[prev_of_removed] = next_of_removed
        prev_idx[next_of_removed] = prev_of_removed

        # update seg_dists for neighbors of removed
        pprev = prev_idx[prev_of_removed]
        nnext = next_idx[next_of_removed]

        pn = [prev_of_removed, next_of_removed]
        left = [pprev, prev_of_removed]
        right = [next_of_removed, nnext]
        
        seg_dists[pn] = line_segment_dists(contour[left],
                                           contour[right],
                                           contour[pn])

    return contour[active_idx]

######################################################################

def main():

    # make output directory
    os.makedirs('shapedata', exist_ok=True)

    # handle inputs
    if len(sys.argv) == 1:
        filenames = sorted(glob.glob('mpeg7_shapes/*-*.gif'))
    else:
        filenames = sys.argv[1:]

    # list to hold (name, shape) pairs
    named_shapes = []

    # for each input file
    for filename in filenames:

        # print it
        print(filename)

        # get name
        name, _ = os.path.splitext(os.path.basename(filename))
        
        # load image using PIL and get size
        image = Image.open(filename).convert('L')
        w, h = image.size

        # make room for border
        enlarged = np.empty((h+2, w+2), dtype=np.uint8)

        image = np.array(image)

        if BLUR_SIGMA:
            image = skimage.filters.gaussian(image, sigma=BLUR_SIGMA, mode='constant', cval=image[0,0])
            image = np.clip(image*255, 0, 255).astype(np.uint8)

        # top left pixel from orig image becomes background color
        enlarged[:] = image[0,0]

        # copy rest of image in
        enlarged[1:h+1, 1:w+1] = image

        # get contours using marching squares
        contours = skimage.measure.find_contours(enlarged, 127.5)

        # areas of each
        areas = np.array([get_bbox_area(contour) for contour in contours])

        # find the contour with largest bounding box - should be outermost
        contour = contours[areas.argmax()]

        # contour should be closed
        assert np.all(contour[-1] == contour[0])

        # omit last point (it's a duplicate)
        contour = contour[:-1]

        #shape = sample_contour_uniformly(contour)
        shape = sample_contour_greedily(contour)

        # swap x & y
        shape = shape[:, ::-1]

        # reverse y
        shape[:, 1] = -shape[:, 1]

        smin = shape.min(axis=0)
        smax = shape.max(axis=0)
        srng = (smax - smin)
        smid = 0.5*(smax+smin)

        # normalize so central coordinate is zero and max abs
        # coordinate is 1
        shape -= smid 
        shape /= srng.max()
        shape += 0.5

        # save
        outfilename = os.path.join('shapedata', name + '.txt')
        np.savetxt(outfilename, shape)

        # stash output for PDF
        named_shapes.append((name, shape))

    if PDF_OUTPUT:
        print()
        print('generating pdf...')
        make_pdf(named_shapes)
        print('done!')
        
    return
        

if __name__ == '__main__':
    main()
